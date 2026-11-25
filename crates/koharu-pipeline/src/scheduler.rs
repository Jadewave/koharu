use std::collections::{BTreeMap, BTreeSet};

use koharu_scene::EntityId;

use crate::Stage;

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    stages: Vec<StageWork>,
    /// The page no longer holds a slot of the sliding window.
    retired: bool,
}

impl PageWork {
    fn started(&self) -> bool {
        self.stages
            .iter()
            .any(|work| work.state != WorkState::Pending)
    }

    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    fn ready(&self, index: usize) -> bool {
        let Some(prerequisite) = prerequisite(self.stages[index].stage) else {
            return true;
        };
        self.stages
            .iter()
            .find(|work| work.stage == prerequisite)
            .is_none_or(|work| work.state == WorkState::Finished)
    }
}

pub(crate) struct Scheduler {
    translation_closed: bool,
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    active_pages: usize,
    head: usize,
    total: usize,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage]) -> Self {
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                stages: stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        state: WorkState::Pending,
                    })
                    .collect(),
                retired: false,
            })
            .collect::<Vec<_>>();
        let total = pages.len().saturating_mul(stages.len());
        Self {
            translation_closed: false,
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: stages.len().max(1),
            active_pages: 0,
            head: 0,
            total,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Holds every translation back until `open_translation` is called, so detection, OCR
    /// and inpainting finish for all pages first. Pages leave the sliding window as soon as
    /// their other stages are done, so the held translations never starve the pages behind.
    pub(crate) fn with_translation_gate(mut self) -> Self {
        self.translation_closed = self.pages.iter().any(|page| {
            page.stages
                .iter()
                .any(|work| work.stage == Stage::Translation)
        });
        self
    }

    pub(crate) fn translation_gated(&self) -> bool {
        self.translation_closed
    }

    pub(crate) fn open_translation(&mut self) {
        self.translation_closed = false;
    }

    pub(crate) fn start_next(
        &mut self,
        busy_stages: &BTreeSet<Stage>,
    ) -> Option<(EntityId, Stage)> {
        for page_index in self.head..self.pages.len() {
            let started = self.pages[page_index].started();
            if !started && self.active_pages >= self.page_window {
                break;
            }
            let stage_index =
                self.pages[page_index]
                    .stages
                    .iter()
                    .enumerate()
                    .find_map(|(index, work)| {
                        (work.state == WorkState::Pending
                            && !busy_stages.contains(&work.stage)
                            && !(self.translation_closed && work.stage == Stage::Translation)
                            && self.pages[page_index].ready(index))
                        .then_some(index)
                    });
            let Some(stage_index) = stage_index else {
                continue;
            };
            if !started {
                self.active_pages += 1;
            }
            let page = &mut self.pages[page_index];
            let work = &mut page.stages[stage_index];
            work.state = WorkState::Running;
            return Some((page.page, work.stage));
        }
        None
    }

    /// Marks a stage finished. Returns true when the page has just left the sliding window
    /// or has just finished entirely, so its cached images can be released.
    pub(crate) fn complete_stage(&mut self, page: EntityId, stage: Stage) -> bool {
        let Some(&page_index) = self.page_index.get(&page) else {
            return false;
        };
        let gated = self.translation_closed;
        let page = &mut self.pages[page_index];
        let was_finished = page.finished();
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Finished;
        }
        let finished = !was_finished && page.finished();
        let settled = if gated {
            page.stages
                .iter()
                .filter(|work| work.stage != Stage::Translation)
                .all(|work| work.state == WorkState::Finished)
        } else {
            page.finished()
        };
        let retired = !page.retired && settled;
        if retired {
            page.retired = true;
            self.active_pages = self.active_pages.saturating_sub(1);
        }
        if finished {
            while self.head < self.pages.len() && self.pages[self.head].finished() {
                self.head += 1;
            }
        }
        retired || finished
    }
}

const fn prerequisite(stage: Stage) -> Option<Stage> {
    match stage {
        Stage::Detection => None,
        Stage::Ocr | Stage::Inpainting => Some(Stage::Detection),
        Stage::Translation => Some(Stage::Ocr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    #[test]
    fn starts_pages_in_order_and_models_independently() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL);
        let mut busy = BTreeSet::new();

        let first = scheduler.start_next(&busy).unwrap();
        assert_eq!(first, (pages[0], Stage::Detection));
        busy.insert(Stage::Detection);
        assert!(scheduler.start_next(&busy).is_none());

        busy.clear();
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);
        let next_page = scheduler.start_next(&busy).unwrap();
        busy.insert(next_page.1);
        assert_eq!(ocr, (pages[0], Stage::Ocr));
        assert_eq!(inpainting, (pages[0], Stage::Inpainting));
        assert_eq!(next_page, (pages[1], Stage::Detection));

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        busy.remove(&Stage::Ocr);
        let translation = scheduler.start_next(&busy).unwrap();
        assert_eq!(translation, (pages[0], Stage::Translation));
        assert!(busy.contains(&Stage::Detection));
        assert!(busy.contains(&Stage::Inpainting));
    }

    #[test]
    fn translation_gate_lets_every_page_finish_its_other_stages_first() {
        let pages = pages(6);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL).with_translation_gate();
        let busy = BTreeSet::new();

        let mut other_stages = 0;
        while let Some((page, stage)) = scheduler.start_next(&busy) {
            assert_ne!(stage, Stage::Translation);
            other_stages += 1;
            scheduler.complete_stage(page, stage);
        }
        assert_eq!(other_stages, 18);
        assert!(scheduler.translation_gated());

        scheduler.open_translation();
        assert!(!scheduler.translation_gated());
        let mut translated = Vec::new();
        while let Some((page, stage)) = scheduler.start_next(&busy) {
            assert_eq!(stage, Stage::Translation);
            translated.push(page);
            assert!(scheduler.complete_stage(page, stage));
        }
        assert_eq!(translated, pages);
    }

    #[test]
    fn translation_gate_is_open_without_a_translation_stage() {
        let pages = pages(2);
        let stages = [Stage::Detection, Stage::Ocr];
        let scheduler = Scheduler::new(&pages, &stages).with_translation_gate();
        assert!(!scheduler.translation_gated());
    }

    #[test]
    fn translation_only_runs_wait_for_the_gate_too() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Translation]).with_translation_gate();
        let busy = BTreeSet::new();
        assert!(scheduler.start_next(&busy).is_none());
        scheduler.open_translation();
        assert_eq!(scheduler.start_next(&busy), Some((pages[0], Stage::Translation)));
        assert!(scheduler.complete_stage(pages[0], Stage::Translation));
        assert_eq!(scheduler.start_next(&busy), Some((pages[1], Stage::Translation)));
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages);
        let mut busy = BTreeSet::new();

        assert_eq!(
            scheduler.start_next(&busy),
            Some((pages[0], Stage::Detection))
        );
        assert!(!scheduler.complete_stage(pages[0], Stage::Detection));
        let ocr = scheduler.start_next(&busy).unwrap();
        busy.insert(ocr.1);
        let inpainting = scheduler.start_next(&busy).unwrap();
        busy.insert(inpainting.1);

        for page in &pages[1..3] {
            assert_eq!(scheduler.start_next(&busy), Some((*page, Stage::Detection)));
            assert!(!scheduler.complete_stage(*page, Stage::Detection));
        }
        assert!(scheduler.start_next(&busy).is_none());

        assert!(!scheduler.complete_stage(pages[0], Stage::Ocr));
        assert!(scheduler.complete_stage(pages[0], Stage::Inpainting));
        busy.clear();
        assert_eq!(scheduler.start_next(&busy), Some((pages[1], Stage::Ocr)));
    }
}

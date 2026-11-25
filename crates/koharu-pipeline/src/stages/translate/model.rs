use koharu_llm::{
    Size,
    rig::{driver::DynModel, operation::Completion},
};

/// A language model's room: one conversation at a time, of text unless
/// translation shows it pages. A local model asks for the most its training
/// allows, and llama.cpp's fit gives it as much of that as the free memory
/// holds, which the loaded model's spec reports; the conversation's budget
/// follows that.
const LANGUAGE: Size = Size {
    sequences: 1,
    context: u32::MAX,
    images: false,
};

/// The model `reference` names, connected, with its projector when it reads
/// `images`; a hosted model has no room to size.
pub(super) async fn load(reference: &str, images: bool) -> anyhow::Result<DynModel<Completion>> {
    koharu_llm::connect(reference, Size { images, ..LANGUAGE }).await
}

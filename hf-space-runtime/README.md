---
title: Koharu Manga Translator
emoji: 📖
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Koharu on Hugging Face Spaces

Runs [Koharu](https://github.com/koharu-rs/koharu) (headless mode,
feat/headless branch, PR #1112) inside a Docker Space.

**This repo's content is managed by GitHub Actions.** The `koharu`
binary here is built on a GitHub-hosted runner (see the source repo's
`.github/workflows/build-and-deploy.yml`) and pushed here automatically
along with this Dockerfile on every build. Do not edit `koharu` by hand
- it will be overwritten on the next deploy.

- First boot will be slow: the app downloads additional ML runtimes and
  models on first run.

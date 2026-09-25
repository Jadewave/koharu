########################################################################
# Deploy Koharu (headless mode, PR #1112 "feat: restore headless
# browser mode", NOT yet merged/released) on a Hugging Face Docker Space
#
# Builds from source on branch koharu-rs/koharu:feat/headless because
# no official release contains this feature yet.
# PR: https://github.com/koharu-rs/koharu/pull/1112
########################################################################

# ---- Stage 1: build ---------------------------------------------------
# Debian bookworm ships gtk4 4.8.3, but Tauri v3-alpha (used by the
# feat/headless branch) needs gtk4 >= 4.14 (gdk4-sys build fails
# otherwise). Debian trixie ships gtk4 4.18, so build on trixie instead.
FROM rust:trixie AS builder

ARG KOHARU_REF=feat/headless
ENV DEBIAN_FRONTEND=noninteractive

# System deps needed to compile a Tauri app + native ML crates.
# Installing both the gtk3/webkit2gtk-4.1 and gtk4/webkit2gtk-6.0 dev
# packages since it's not yet confirmed which one this Tauri v3-alpha
# branch links against - extras are harmless, just extra apt time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates pkg-config build-essential \
    libwebkit2gtk-4.1-dev \
    libwebkitgtk-6.0-dev \
    libgtk-3-dev \
    libgtk-4-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libssl-dev \
    libnss3-dev \
    libasound2-dev \
    libx11-xcb-dev \
    libxcomposite-dev \
    libxdamage-dev \
    libxrandr-dev \
    libxkbcommon-dev \
    file \
    && rm -rf /var/lib/apt/lists/*

# Bun (needed for the frontend / bridge packages)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

WORKDIR /src
RUN git clone --depth 1 --branch ${KOHARU_REF} \
    https://github.com/koharu-rs/koharu.git .

# The @koharu/bridge build script (wasm-pack) resolves crate paths as if
# the repo root were "/", e.g. "/crates/koharu-canvas". Symlink the crates
# dir AND the workspace manifests so cargo can find the workspace root when
# walking up from /crates/<crate>/Cargo.toml.
RUN ln -s /src/crates /crates \
 && ln -s /src/Cargo.toml /Cargo.toml \
 && ln -s /src/Cargo.lock /Cargo.lock

# Install JS deps and build the bridge + frontend before compiling Rust,
# since the headless server embeds the pre-built frontend.
RUN bun install
RUN bun run --filter @koharu/bridge build

# `@koharu/bridge build` only regenerates the TS protocol bindings. The WASM
# module the frontend imports (packages/bridge/src/wasm/koharu_canvas.js) is
# normally produced by the "bun dev" watcher via wasm-pack, which never runs
# during a one-shot "build". Build it explicitly here, using the "web"
# wasm-bindgen target (bridge code does `await canvas.default()` on import,
# which is the "web" target's manual-init pattern).
RUN cargo install wasm-pack --locked
RUN wasm-pack build crates/koharu-canvas \
    --target web \
    --out-dir /src/packages/bridge/src/wasm \
    --out-name koharu_canvas

RUN bun run --filter @koharu/app build

# Build the Rust workspace in release mode.
# NOTE: the exact package name that produces the `koharu` binary is
# unconfirmed (workspace may be split into koharu-app / koharu-core /
# koharu-macros). Try the whole-workspace build first; if it errors
# about ambiguous binaries, replace with `-p koharu-app` or similar
# per the actual error message.
RUN cargo build --release
RUN find /src/target/release -maxdepth 1 -type f -executable -name 'koharu*' \
    -exec cp {} /usr/local/bin/koharu \; \
    && test -x /usr/local/bin/koharu

# ---- Stage 2: runtime ---------------------------------------------------
# Match the builder's distro (Debian trixie) so the runtime linker finds
# libraries at the same glibc/gtk4 ABI the binary was built against.
# NOTE: Debian package names here (no Ubuntu "t64" suffix renaming).
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/user \
    RUST_LOG=info

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl socat \
    libwebkit2gtk-4.1-0 \
    libwebkitgtk-6.0-4 \
    libgtk-3-0 \
    libgtk-4-1 \
    libayatana-appindicator3-1 \
    librsvg2-2 \
    libssl3 \
    libnss3 \
    libasound2 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxkbcommon0 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/koharu /usr/local/bin/koharu

RUN useradd -m -u 1001 user \
    && mkdir -p /home/user/.local/share /home/user/.cache \
    && chown -R user:user /home/user
USER user
WORKDIR /home/user

EXPOSE 7860

# PR #1112 defaults to binding 127.0.0.1, so bind 0.0.0.0 explicitly to
# be reachable from outside the container on HF Spaces.
CMD ["koharu", "--headless", "--host", "0.0.0.0", "--port", "7860"]

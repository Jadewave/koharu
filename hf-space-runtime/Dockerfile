########################################################################
# Runtime-only image for the Koharu HF Space.
#
# The heavy work (cargo build --release of the Tauri app, wasm-pack,
# Next.js build) happens in GitHub Actions, NOT here - see
# .github/workflows/build-and-deploy.yml in the source repo. That
# workflow pushes this Dockerfile + the compiled `koharu` binary here
# together, so the Space just has to COPY the binary in and install
# runtime shared libraries. This build should finish in well under a
# minute, even on a free CPU-basic Space.
########################################################################
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

# Pushed alongside this Dockerfile by the GitHub Actions deploy job.
COPY koharu /usr/local/bin/koharu
RUN chmod +x /usr/local/bin/koharu

RUN useradd -m -u 1001 user \
    && mkdir -p /home/user/.local/share /home/user/.cache \
    && chown -R user:user /home/user
USER user
WORKDIR /home/user

EXPOSE 7860

# PR #1112 defaults to binding 127.0.0.1, so bind 0.0.0.0 explicitly to
# be reachable from outside the container on HF Spaces.
CMD ["koharu", "--headless", "--host", "0.0.0.0", "--port", "7860"]

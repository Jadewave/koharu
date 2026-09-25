########################################################################
# Runtime-only image for the Koharu HF Space.
#
# The heavy work (cargo build --release of the Tauri app, wasm-pack,
# Next.js build) happens in GitHub Actions, NOT here - see
# .github/workflows/build-and-deploy.yml in the source repo.
#
# The compiled `koharu` binary + CEF (Chromium Embedded Framework)
# runtime files are NOT committed to this repo: libcef.so alone is
# ~1.3GB, which blows through the HF Space free tier's 1GB git/LFS
# storage cap on its own. Instead they're published as a GitHub Release
# asset by that workflow, and downloaded here at Docker-build time -
# this repo (and its git history) stays tiny, only the release tarball
# needs the bigger transfer, and that happens fresh on every build.
#
# __RUNTIME_TARBALL_URL__ below is filled in automatically by the
# deploy job with this repo's own GitHub Release URL - no manual edit
# needed.
########################################################################
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/user \
    RUST_LOG=info \
    CEF_PATH=/opt/cef \
    LD_LIBRARY_PATH=/opt/cef

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

# Downloads koharu-runtime.tar.gz (koharu binary + cef/ directory) from
# this fork's GitHub Release and unpacks it - see header comment above.
RUN curl -fsSL "__RUNTIME_TARBALL_URL__" -o /tmp/koharu-runtime.tar.gz \
    && mkdir -p /opt/koharu-runtime \
    && tar xzf /tmp/koharu-runtime.tar.gz -C /opt/koharu-runtime \
    && mv /opt/koharu-runtime/koharu /usr/local/bin/koharu \
    && chmod +x /usr/local/bin/koharu \
    && mv /opt/koharu-runtime/cef /opt/cef \
    && rm -rf /opt/koharu-runtime /tmp/koharu-runtime.tar.gz

RUN useradd -m -u 1001 user \
    && mkdir -p /home/user/.local/share /home/user/.cache \
    && chown -R user:user /home/user /opt/cef
USER user
WORKDIR /home/user

EXPOSE 7860

# PR #1112 defaults to binding 127.0.0.1, so bind 0.0.0.0 explicitly to
# be reachable from outside the container on HF Spaces.
CMD ["koharu", "--headless", "--host", "0.0.0.0", "--port", "7860"]

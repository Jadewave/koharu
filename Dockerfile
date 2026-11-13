# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/root/.bun/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang cmake curl git \
    libclang-dev libclang-18-dev libcairo2-dev libgdk-pixbuf-2.0-dev \
    libgtk-4-dev libpango1.0-dev libssl-dev llvm-dev pkg-config \
    python3 unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /src
COPY . .

RUN cargo install tauri-cli --version '=3.0.0-alpha.1' --locked

ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib

RUN bun install --frozen-lockfile
RUN bun run build

# ONNX Runtime for PP-OCRv6 manga. The Rust side loads it at run time (`ort`
# `load-dynamic`), so nothing is linked at build time. Keep the version in step
# with the `ort` release pinned in Cargo.toml (rc.10 expects ONNX Runtime 1.22.x).
ARG ORT_VERSION=1.22.0
RUN mkdir -p /opt/ort \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
        | tar -xz -C /tmp \
    && cp -L "/tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" /opt/ort/libonnxruntime.so \
    && rm -rf "/tmp/onnxruntime-linux-x64-${ORT_VERSION}"

# The CEF runtime (libcef.so and its resources) is not always next to the binary in
# target/release, so look for it where the build left it and gather what the app needs in
# /opt/cef. When it is nowhere, say so and show what the build produced.
RUN set -eux; \
    mkdir -p /opt/cef; \
    lib="$(find /src/target /root -name libcef.so -type f 2>/dev/null | head -n 1 || true)"; \
    if [ -z "$lib" ]; then \
        echo "libcef.so was not produced by the build. target/release holds:"; \
        ls -la /src/target/release | head -n 80; \
        find / -xdev \( -iname 'libcef*' -o -iname 'icudtl.dat' \) 2>/dev/null | head -n 40; \
        exit 1; \
    fi; \
    dir="$(dirname "$lib")"; \
    echo "CEF runtime found in $dir"; \
    ls -la "$dir" | head -n 80; \
    for f in libcef.so icudtl.dat resources.pak locales chrome-sandbox \
             chrome_100_percent.pak chrome_200_percent.pak \
             v8_context_snapshot.bin snapshot_blob.bin \
             libEGL.so libGLESv2.so libvk_swiftshader.so libvulkan.so.1 vk_swiftshader_icd.json; do \
        if [ -e "$dir/$f" ]; then cp -a "$dir/$f" /opt/cef/; fi; \
    done; \
    test -f /opt/cef/libcef.so; \
    test -f /opt/cef/icudtl.dat; \
    test -f /opt/cef/resources.pak; \
    ls -la /opt/cef


FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libasound2t64 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libatspi2.0-0 \
    libcairo2 \
    libdbus-1-3 \
    libdrm2 \
    libexpat1 \
    libfontconfig1 \
    libfreetype6 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-4-1 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-dri3-0 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxtst6 \
    xdg-user-dirs \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
    /home/user/Documents \
    /home/user/Desktop \
    /home/user/Downloads \
    /home/user/Pictures \
    /home/user/Music \
    /home/user/Videos \
    /home/user/.config \
    /home/user/.local/share \
    /home/user/.cache \
    && chown -R 1000:1000 /home/user

ENV HOME=/home/user
ENV XDG_CONFIG_HOME=/home/user/.config
ENV XDG_DATA_HOME=/home/user/.local/share
ENV XDG_CACHE_HOME=/home/user/.cache

RUN chown -R 1000:1000 /home/user \
    && su -s /bin/sh -c 'xdg-user-dirs-update' "$(getent passwd 1000 | cut -d: -f1)"

WORKDIR /app

COPY --from=builder /src/target/release/koharu /app/koharu
COPY --from=builder /opt/cef/ /app/
COPY --from=builder /opt/ort/libonnxruntime.so /app/libonnxruntime.so

RUN chown -R 1000:1000 /app \
    && if [ -f /app/chrome-sandbox ]; then \
        chown root:root /app/chrome-sandbox && chmod 4755 /app/chrome-sandbox; \
    fi

ENV LD_LIBRARY_PATH=/app
ENV WGPU_BACKEND=vulkan

EXPOSE 7860

ENTRYPOINT ["/app/koharu"]

CMD ["--headless", "--host", "0.0.0.0", "--port", "7860", "--cpu"]

# syntax=docker/dockerfile:1
#
# Koharu's pipeline behind HTTP (crates/koharu-server), for a container with no
# screen, such as a Hugging Face Docker Space. There's no Tauri, CEF or GTK in
# it: the page it serves is one file embedded in the binary.

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang cmake curl git \
    libclang-dev libclang-18-dev libssl-dev libfontconfig1-dev libfreetype-dev \
    llvm-dev pkg-config python3 python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable

ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib

# ONNX Runtime for PP-OCRv6 manga. The Rust side loads it at run time (`ort`
# `load-dynamic`), so nothing is linked at build time. Keep the version in step
# with the `ort` release pinned in Cargo.toml (rc.10 expects ONNX Runtime 1.22.x).
ARG ORT_VERSION=1.22.0
# Primary source: the official GitHub release tarball (downloaded to a file
# first so a 404 fails loudly instead of feeding an empty stream to tar).
# Fallback: the PyPI wheel, which ships the very same libonnxruntime.so.
RUN set -eux; \
    mkdir -p /opt/ort /tmp/ort; \
    if curl -fsSL --retry 3 --retry-delay 2 \
        -o /tmp/ort/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
       && tar -xzf /tmp/ort/ort.tgz -C /tmp/ort; then \
        cp -L "/tmp/ort/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" /opt/ort/libonnxruntime.so; \
    else \
        echo "GitHub download failed, falling back to the PyPI wheel"; \
        pip3 download "onnxruntime==${ORT_VERSION}" --no-deps --only-binary=:all: \
            --break-system-packages -d /tmp/ort/whl; \
        unzip -p /tmp/ort/whl/onnxruntime-*.whl "onnxruntime/capi/libonnxruntime.so.${ORT_VERSION}" \
            > /opt/ort/libonnxruntime.so; \
    fi; \
    test -s /opt/ort/libonnxruntime.so; \
    rm -rf /tmp/ort

WORKDIR /src
COPY . .
RUN cargo build --release --locked -p koharu-server

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# libvulkan1 and mesa-vulkan-drivers (lavapipe) give the renderer a Vulkan
# device on a machine with no GPU; the fonts are for lettering the page.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libfontconfig1 libfreetype6 libgomp1 libssl3 \
    libvulkan1 mesa-vulkan-drivers fonts-dejavu-core fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs a Space as user 1000.
RUN mkdir -p /home/user/.koharu /home/user/.cache && chown -R 1000:1000 /home/user
ENV HOME=/home/user
ENV XDG_CACHE_HOME=/home/user/.cache

WORKDIR /app
COPY --from=builder /src/target/release/koharu-server /app/koharu-server
COPY --from=builder /opt/ort/libonnxruntime.so /app/libonnxruntime.so
RUN chown -R 1000:1000 /app

ENV LD_LIBRARY_PATH=/app
ENV ORT_DYLIB_PATH=/app/libonnxruntime.so
ENV WGPU_BACKEND=vulkan
USER 1000

EXPOSE 7860
ENTRYPOINT ["/app/koharu-server"]
CMD ["--host", "0.0.0.0", "--port", "7860"]

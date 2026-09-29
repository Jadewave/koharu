# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/root/.bun/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libasound2t64 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libatspi2.0-0 \
    libcairo2 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-4-1 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
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
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxtst6 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libexpat1 \
    libnspr4 \
    libwayland-client0 \
    libwayland-egl1 \
    libwayland-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /src

COPY . .

# Install Tauri CLI required by the project
RUN cargo install tauri-cli --version '=3.0.0-alpha.1' --locked

ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib

# Install JavaScript dependencies
RUN bun install --frozen-lockfile

# Build WASM, frontend and Koharu
RUN bun run build


FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libasound2t64 \
    libcairo2 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-4-1 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
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
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxtst6 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libexpat1 \
    libnspr4 \
    libwayland-client0 \
    libwayland-egl1 \
    libwayland-cursor0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Koharu executable
COPY --from=builder /src/target/release/koharu /app/koharu

# CEF runtime
COPY --from=builder /src/target/release/libcef.so /app/libcef.so
COPY --from=builder /src/target/release/icudtl.dat /app/icudtl.dat
COPY --from=builder /src/target/release/resources.pak /app/resources.pak
COPY --from=builder /src/target/release/locales /app/locales
COPY --from=builder /src/target/release/chrome-sandbox /app/chrome-sandbox

# CEF sandbox requires the setuid bit
RUN chmod 4755 /app/chrome-sandbox

ENV LD_LIBRARY_PATH=/app

EXPOSE 7860

ENTRYPOINT ["/app/koharu"]

CMD ["--headless", "--host", "0.0.0.0", "--port", "7860", "--cpu"]

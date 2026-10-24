use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::pp_ocr_v6_manga::PpOcrV6Manga;

#[derive(Debug, Parser)]
#[command(about = "Run PP-OCRv6 manga text detection and recognition on a text-block crop")]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(cli.input)?;

    // ONNX Runtime needs no process-wide initialization, so `koharu_ml::init` is not called.
    let model = PpOcrV6Manga::load(koharu_ml::Device::cpu()).await?;
    println!("{}", model.inference(&image)?);
    Ok(())
}

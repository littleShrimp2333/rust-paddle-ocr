//! OCR CLI — PP-OCR 命令行工具
//!
//! 支持 PP-OCRv4 / v5 / v6 模型的文本检测与识别。
//!
//! 用法 (通用模式):
//!   ocr-cli <image> [选项]
//!
//! 示例:
//!   ocr-cli photo.jpg                     # 默认 v5
//!   ocr-cli photo.jpg -m v6               # 使用 v6 模型
//!   ocr-cli photo.jpg -m v4               # 使用 v4 模型
//!   ocr-cli photo.jpg --json
//!
//! 内嵌模式 (模型编译进二进制, 无需外部模型文件):
//!   cargo build --release --features bundle-models-v6
//!   ./ocr-cli photo.jpg                   # 自带 v6 模型, 不用 -m 参数

use ocr_rs::{Backend, OcrEngine, OcrEngineConfig};
use std::env;
use std::path::{Path, PathBuf};
use std::process;

// ── 内嵌模型数据 ──────────────────────────────────────────

#[cfg(feature = "bundle-models-v4")]
const EMBEDDED_VERSION: &str = "v4";
#[cfg(feature = "bundle-models-v5")]
const EMBEDDED_VERSION: &str = "v5";
#[cfg(feature = "bundle-models-v6")]
const EMBEDDED_VERSION: &str = "v6";

#[cfg(not(any(
    feature = "bundle-models-v4",
    feature = "bundle-models-v5",
    feature = "bundle-models-v6",
)))]
const EMBEDDED_VERSION: &str = "";

// 编译期检查: 不能同时启用多个 bundle-models-*
#[cfg(any(
    all(feature = "bundle-models-v4", feature = "bundle-models-v5"),
    all(feature = "bundle-models-v4", feature = "bundle-models-v6"),
    all(feature = "bundle-models-v5", feature = "bundle-models-v6"),
))]
compile_error!("只能启用一个 bundle-models-* feature (v4 / v5 / v6 三选一)");

/// 默认模型目录
const DEFAULT_MODEL_DIR: &str = "models";

/// 各版本模型路径
struct ModelSet {
    det: &'static str,
    rec: &'static str,
    keys: &'static str,
}

const MODEL_V4: ModelSet = ModelSet {
    det: "ch_PP-OCRv4_det_infer.mnn",
    rec: "ch_PP-OCRv4_rec_infer.mnn",
    keys: "ppocr_keys_v4.txt",
};

const MODEL_V5: ModelSet = ModelSet {
    det: "PP-OCRv5_mobile_det.mnn",
    rec: "PP-OCRv5_mobile_rec.mnn",
    keys: "ppocr_keys_v5.txt",
};

const MODEL_V6: ModelSet = ModelSet {
    det: "PP-OCRv6_small_det.mnn",
    rec: "PP-OCRv6_small_rec.mnn",
    keys: "ppocr_keys_v6.txt",
};

const HELP: &str = r#"OCR CLI — PaddleOCR 命令行工具

用法: ocr-cli <image> [选项]

参数:
  <image>              要识别的图片路径

选项:
  --path <image>       图片路径 (兼容模式，等价于位置参数)
  -m, --model <ver>    选择模型版本: v4, v5, v6 (默认: v5)
                       -m json 等价于 --json (输出 JSON 格式)
  --det <path>         检测模型路径 (覆盖 --model 的默认值)
  --rec <path>         识别模型路径 (覆盖 --model 的默认值)
  --keys <path>        字符集文件路径 (覆盖 --model 的默认值)
  --ori <path>         方向分类模型路径 (可选)
  --output <path>      输出图片路径 (标注文字框后保存)
  --backend <name>     推理后端: cpu, metal, opencl, vulkan, cuda, opengl (默认: cpu)
  --threads <n>        线程数 (默认: 自动)
  --json               以 JSON 格式输出结果
  --quiet              静默模式，不输出进度信息
  --list-models        列出 models/ 目录中可用的模型文件
  --help, -h           显示此帮助信息
  --version, -V        显示版本信息

示例:
  ocr-cli photo.jpg                     # v5 默认
  ocr-cli photo.jpg -m v6               # v6 模型
  ocr-cli photo.jpg -m v4               # v4 模型
  ocr-cli photo.jpg --json
  ocr-cli photo.jpg -m json             # JSON 输出 (兼容模式)
  ocr-cli --path photo.jpg -m json      # 兼容外部调用
  ocr-cli photo.jpg -m v6 --backend metal --output annotated.png
  ocr-cli --list-models
"#;

struct CliArgs {
    image: PathBuf,
    det_model: Option<PathBuf>,
    rec_model: Option<PathBuf>,
    keys: Option<PathBuf>,
    ori_model: Option<PathBuf>,
    output: Option<PathBuf>,
    backend: Backend,
    threads: Option<u32>,
    json: bool,
    quiet: bool,
    list_models: bool,
    help: bool,
    version: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("错误: {e}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    if args.help {
        print!("{HELP}");
        return Ok(());
    }

    if args.version {
        println!("ocr-cli v{} (MNN: {})", ocr_rs::version(), ocr_rs::mnn_version());
        return Ok(());
    }

    if args.list_models {
        return list_models();
    }

    let mut config = OcrEngineConfig::new().with_backend(args.backend);
    if let Some(n) = args.threads {
        config = config.with_threads(n as i32);
    }

    // 创建引擎: 内嵌模式 vs 文件模式
    let engine = if !EMBEDDED_VERSION.is_empty() {
        create_embedded_engine(config, args.json || args.quiet)?
    } else {
        create_file_engine(&args, config)?
    };

    if !args.quiet && !args.json {
        eprintln!("   图片: {}", args.image.display());
    }

    let image = image::open(&args.image)
        .map_err(|e| format!("无法打开图片 '{}': {e}", args.image.display()))?;

    let results = engine.recognize(&image)?;

    if args.json {
        output_json(&results, args.json || args.quiet)?;
    } else {
        output_text(&results, args.quiet);
    }

    if let Some(ref output_path) = args.output {
        save_annotated_image(&image, &results, output_path)?;
        if !args.quiet && !args.json {
            eprintln!("   ✅ 标注图片已保存: {}", output_path.display());
        }
    }

    Ok(())
}

/// 内嵌模式: 模型数据编译进二进制
fn create_embedded_engine(config: OcrEngineConfig, quiet: bool) -> Result<OcrEngine, Box<dyn std::error::Error>> {
    let (det_data, rec_data, keys_data) = get_embedded_models();
    if !quiet {
        eprintln!("🔍 OCR CLI v{} [内嵌 {} 模型]", ocr_rs::version(), EMBEDDED_VERSION);
    }
    Ok(OcrEngine::from_bytes(det_data, rec_data, keys_data, Some(config))?)
}

/// 文件模式: 从路径加载模型
fn create_file_engine(args: &CliArgs, config: OcrEngineConfig) -> Result<OcrEngine, Box<dyn std::error::Error>> {
    let det = args.det_model.as_ref().expect("缺少检测模型路径");
    let rec = args.rec_model.as_ref().expect("缺少识别模型路径");
    let keys = args.keys.as_ref().expect("缺少字符集路径");

    if !args.quiet && !args.json {
        eprintln!("🔍 OCR CLI v{}", ocr_rs::version());
        eprintln!("   检测: {}", det.display());
        eprintln!("   识别: {}", rec.display());
        eprintln!("   字符集: {}", keys.display());
        if let Some(ref ori) = args.ori_model {
            eprintln!("   方向: {}", ori.display());
        }
        eprintln!("   后端: {:?}", args.backend);
    }

    if let Some(ref ori_path) = args.ori_model {
        Ok(OcrEngine::new_with_ori(det, rec, keys, ori_path, Some(config))?)
    } else {
        Ok(OcrEngine::new(det, rec, keys, Some(config))?)
    }
}

// ── 内嵌模型数据 (include_bytes!) ────────────────────────

#[cfg(feature = "bundle-models-v4")]
fn get_embedded_models() -> (&'static [u8], &'static [u8], &'static [u8]) {
    (
        include_bytes!("../models/ch_PP-OCRv4_det_infer.mnn"),
        include_bytes!("../models/ch_PP-OCRv4_rec_infer.mnn"),
        include_bytes!("../models/ppocr_keys_v4.txt"),
    )
}

#[cfg(feature = "bundle-models-v5")]
fn get_embedded_models() -> (&'static [u8], &'static [u8], &'static [u8]) {
    (
        include_bytes!("../models/PP-OCRv5_mobile_det.mnn"),
        include_bytes!("../models/PP-OCRv5_mobile_rec.mnn"),
        include_bytes!("../models/ppocr_keys_v5.txt"),
    )
}

#[cfg(feature = "bundle-models-v6")]
fn get_embedded_models() -> (&'static [u8], &'static [u8], &'static [u8]) {
    (
        include_bytes!("../models/PP-OCRv6_small_det.mnn"),
        include_bytes!("../models/PP-OCRv6_small_rec.mnn"),
        include_bytes!("../models/ppocr_keys_v6.txt"),
    )
}

#[cfg(not(any(
    feature = "bundle-models-v4",
    feature = "bundle-models-v5",
    feature = "bundle-models-v6",
)))]
fn get_embedded_models() -> (&'static [u8], &'static [u8], &'static [u8]) {
    // 不会调用到 — 只是让编译器开心
    (&[], &[], &[])
}

// ── 命令行解析 ───────────────────────────────────────────

fn parse_args() -> CliArgs {
    let raw: Vec<String> = env::args().collect();
    if raw.len() < 2 {
        print!("{HELP}");
        process::exit(0);
    }
    let args = parse_args_impl(&raw);
    if args.image.as_os_str().is_empty() && !args.help && !args.version && !args.list_models {
        eprintln!("错误: 缺少图片参数");
        eprintln!("用法: ocr-cli <image> [选项]");
        eprintln!("使用 --help 查看完整帮助");
        process::exit(1);
    }
    args
}

/// 纯解析逻辑，不调用 process::exit，方便单元测试
fn parse_args_impl(raw: &[String]) -> CliArgs {
    let is_bundled = !EMBEDDED_VERSION.is_empty();

    let model_set = &MODEL_V5;

    let mut args = CliArgs {
        image: PathBuf::new(),
        det_model: if is_bundled {
            None
        } else {
            Some(PathBuf::from(DEFAULT_MODEL_DIR).join(model_set.det))
        },
        rec_model: if is_bundled {
            None
        } else {
            Some(PathBuf::from(DEFAULT_MODEL_DIR).join(model_set.rec))
        },
        keys: if is_bundled {
            None
        } else {
            Some(PathBuf::from(DEFAULT_MODEL_DIR).join(model_set.keys))
        },
        ori_model: None,
        output: None,
        backend: Backend::CPU,
        threads: None,
        json: false,
        quiet: false,
        list_models: false,
        help: false,
        version: false,
    };

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" => args.help = true,
            "-V" | "--version" => args.version = true,
            "--list-models" => args.list_models = true,
            "--json" => args.json = true,
            "--quiet" => args.quiet = true,
            "-m" | "--model" => {
                i += 1;
                let val = raw.get(i).cloned().unwrap_or_default();
                // 兼容 -m json 作为 --json 的别名
                if val == "json" {
                    args.json = true;
                } else {
                    if is_bundled {
                        eprintln!("警告: 内嵌模式已固定为 {} 模型，--model 参数被忽略", EMBEDDED_VERSION);
                    } else {
                        let ms = match val.as_str() {
                            "v4" => &MODEL_V4,
                            "v5" => &MODEL_V5,
                            "v6" => &MODEL_V6,
                            other => {
                                eprintln!("警告: 未知模型版本 '{:?}'，使用 v5", other);
                                &MODEL_V5
                            }
                        };
                        args.det_model = Some(PathBuf::from(DEFAULT_MODEL_DIR).join(ms.det));
                        args.rec_model = Some(PathBuf::from(DEFAULT_MODEL_DIR).join(ms.rec));
                        args.keys = Some(PathBuf::from(DEFAULT_MODEL_DIR).join(ms.keys));
                    }
                }
            }
            "--path" => {
                i += 1;
                args.image = PathBuf::from(raw.get(i).cloned().unwrap_or_default());
            }
            "--det" => {
                i += 1;
                if is_bundled {
                    eprintln!("警告: 内嵌模式已包含模型，--det 参数被忽略");
                } else {
                    args.det_model = Some(PathBuf::from(raw.get(i).cloned().unwrap_or_default()));
                }
            }
            "--rec" => {
                i += 1;
                if is_bundled {
                    eprintln!("警告: 内嵌模式已包含模型，--rec 参数被忽略");
                } else {
                    args.rec_model = Some(PathBuf::from(raw.get(i).cloned().unwrap_or_default()));
                }
            }
            "--keys" => {
                i += 1;
                if is_bundled {
                    eprintln!("警告: 内嵌模式已包含字符集，--keys 参数被忽略");
                } else {
                    args.keys = Some(PathBuf::from(raw.get(i).cloned().unwrap_or_default()));
                }
            }
            "--ori" => {
                i += 1;
                args.ori_model = Some(PathBuf::from(raw.get(i).cloned().unwrap_or_default()));
            }
            "--output" => {
                i += 1;
                args.output = Some(PathBuf::from(raw.get(i).cloned().unwrap_or_default()));
            }
            "--backend" => {
                i += 1;
                args.backend = parse_backend(raw.get(i).cloned().unwrap_or_default().as_str());
            }
            "--threads" => {
                i += 1;
                args.threads = raw.get(i).cloned().unwrap_or_default().parse().ok();
            }
            arg if !arg.starts_with('-') && args.image.as_os_str().is_empty() => {
                args.image = PathBuf::from(arg);
            }
            _ => {
                eprintln!("警告: 未知参数 '{}'，已忽略", raw[i]);
            }
        }
        i += 1;
    }

    args
}

fn parse_backend(s: &str) -> Backend {
    match s.to_lowercase().as_str() {
        "cpu" => Backend::CPU,
        "metal" => Backend::Metal,
        "opencl" => Backend::OpenCL,
        "opengl" => Backend::OpenGL,
        "vulkan" => Backend::Vulkan,
        "cuda" => Backend::CUDA,
        _ => {
            eprintln!("警告: 未知后端 '{}'，使用 CPU", s);
            Backend::CPU
        }
    }
}

// ── 输出 ─────────────────────────────────────────────────

fn output_text(results: &[ocr_rs::OcrResult_], quiet: bool) {
    if !quiet {
        eprintln!("\n📝 识别到 {} 个文本区域:\n", results.len());
    }

    for (i, r) in results.iter().enumerate() {
        let bbox = &r.bbox;
        println!(
            "[{:2}] {:6.2}%  ({:4},{:4}) {:4}x{:<4}  {}",
            i + 1,
            r.confidence * 100.0,
            bbox.rect.left(),
            bbox.rect.top(),
            bbox.rect.width(),
            bbox.rect.height(),
            r.text,
        );
    }

    if !quiet && !results.is_empty() {
        let avg = results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
        println!("\n共 {} 个区域 | 平均置信度: {:.2}%", results.len(), avg * 100.0);
    }
}

fn output_json(results: &[ocr_rs::OcrResult_], quiet: bool) -> Result<(), Box<dyn std::error::Error>> {
    let avg = if results.is_empty() {
        0.0
    } else {
        (results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32 * 1000.0).round()
            / 1000.0
    };

    print!(
        "{{\n  \"count\": {},\n  \"avg_confidence\": {:.3},\n  \"results\": [\n",
        results.len(),
        avg
    );

    for (i, r) in results.iter().enumerate() {
        let bbox = &r.bbox;
        let comma = if i + 1 < results.len() { "," } else { "" };
        print!("    {{\n      \"text\": {:?},\n", r.text);
        print!(
            "      \"confidence\": {:.3},\n",
            (r.confidence * 1000.0).round() / 1000.0
        );
        print!(
            "      \"box\": {{\"left\": {}, \"top\": {}, \"width\": {}, \"height\": {}}}",
            bbox.rect.left(),
            bbox.rect.top(),
            bbox.rect.width(),
            bbox.rect.height(),
        );
        if let Some(ref pts) = bbox.points {
            print!(",\n      \"points\": [");
            for (j, p) in pts.iter().enumerate() {
                if j > 0 {
                    print!(", ");
                }
                print!("{{\"x\": {}, \"y\": {}}}", p.x, p.y);
            }
            print!("]");
        }
        println!("\n    }}{}", comma);
    }

    println!("  ]\n}}");

    if !quiet {
        eprintln!("\n✅ 已输出 {} 个结果 (JSON)", results.len());
    }
    Ok(())
}

fn save_annotated_image(
    image: &image::DynamicImage,
    results: &[ocr_rs::OcrResult_],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use image::Rgb;
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;

    let mut output_image = image.to_rgb8();

    let colors = [
        Rgb([255u8, 0, 0]),
        Rgb([0, 255, 0]),
        Rgb([0, 0, 255]),
        Rgb([255, 255, 0]),
        Rgb([255, 0, 255]),
        Rgb([0, 255, 255]),
        Rgb([255, 128, 0]),
        Rgb([128, 0, 255]),
    ];

    for (i, result) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        let bbox = &result.bbox;
        let rect = Rect::at(bbox.rect.left(), bbox.rect.top())
            .of_size(bbox.rect.width(), bbox.rect.height());
        draw_hollow_rect_mut(&mut output_image, rect, color);
        if bbox.rect.left() > 0 && bbox.rect.top() > 0 {
            let rect2 = Rect::at(bbox.rect.left() - 1, bbox.rect.top() - 1)
                .of_size(bbox.rect.width() + 2, bbox.rect.height() + 2);
            draw_hollow_rect_mut(&mut output_image, rect2, color);
        }
    }

    output_image.save(output_path)?;
    Ok(())
}

/// 列出 models/ 目录中的可用模型
fn list_models() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new(DEFAULT_MODEL_DIR);
    if !model_dir.exists() {
        println!("models/ 目录不存在");
        return Ok(());
    }

    println!("📦 models/ 目录中的可用文件:\n");

    let mut entries: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    entries.sort();

    let det_files: Vec<_> = entries.iter().filter(|n| n.contains("det")).collect();
    let rec_files: Vec<_> = entries.iter().filter(|n| n.contains("rec")).collect();
    let keys_files: Vec<_> = entries.iter().filter(|n| n.starts_with("ppocr_keys")).collect();
    let ori_files: Vec<_> = entries
        .iter()
        .filter(|n| n.contains("ori") || n.contains("LCNet"))
        .collect();
    let other_files: Vec<_> = entries
        .iter()
        .filter(|n| {
            !n.contains("det")
                && !n.contains("rec")
                && !n.starts_with("ppocr_keys")
                && !(n.contains("ori") || n.contains("LCNet"))
        })
        .collect();

    if !det_files.is_empty() {
        println!("  检测模型 (Detection):");
        for f in &det_files {
            println!("    {}", f);
        }
        println!();
    }
    if !rec_files.is_empty() {
        println!("  识别模型 (Recognition):");
        for f in &rec_files {
            println!("    {}", f);
        }
        println!();
    }
    if !keys_files.is_empty() {
        println!("  字符集 (Charset):");
        for f in &keys_files {
            println!("    {}", f);
        }
        println!();
    }
    if !ori_files.is_empty() {
        println!("  方向模型 (Orientation):");
        for f in &ori_files {
            println!("    {}", f);
        }
        println!();
    }
    if !other_files.is_empty() {
        println!("  其他文件:");
        for f in &other_files {
            println!("    {}", f);
        }
        println!();
    }

    let versions: Vec<&str> = entries
        .iter()
        .filter_map(|n| {
            if n.contains("v4") {
                Some("v4")
            } else if n.contains("v5") {
                Some("v5")
            } else if n.contains("v6") {
                Some("v6")
            } else {
                None
            }
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    if !versions.is_empty() {
        println!("  检测到的模型版本: {}", versions.join(", "));
        println!();
        println!("  用法: ocr-cli <image> -m <版本>");
        for v in &versions {
            println!("    ocr-cli photo.jpg -m {}", v);
        }
    }

    Ok(())
}

// ============================================================
// 单元测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ──────────────────────────────────────────

    /// 构建模拟的 argv: ["ocr-cli", ...rest]
    fn argv(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["ocr-cli".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    // ============================================================
    // parse_backend 测试
    // ============================================================

    #[test]
    fn test_parse_backend_cpu() {
        assert!(matches!(parse_backend("cpu"), Backend::CPU));
        assert!(matches!(parse_backend("CPU"), Backend::CPU));
    }

    #[test]
    fn test_parse_backend_metal() {
        assert!(matches!(parse_backend("metal"), Backend::Metal));
    }

    #[test]
    fn test_parse_backend_opencl() {
        assert!(matches!(parse_backend("opencl"), Backend::OpenCL));
    }

    #[test]
    fn test_parse_backend_opengl() {
        assert!(matches!(parse_backend("opengl"), Backend::OpenGL));
    }

    #[test]
    fn test_parse_backend_vulkan() {
        assert!(matches!(parse_backend("vulkan"), Backend::Vulkan));
    }

    #[test]
    fn test_parse_backend_cuda() {
        assert!(matches!(parse_backend("cuda"), Backend::CUDA));
    }

    #[test]
    fn test_parse_backend_unknown_fallback() {
        assert!(matches!(parse_backend("nonsense"), Backend::CPU));
        assert!(matches!(parse_backend(""), Backend::CPU));
    }

    // ============================================================
    // parse_args_impl 基础标志测试
    // ============================================================

    #[test]
    fn test_help_short() {
        let a = argv(&["-h"]);
        let args = parse_args_impl(&a);
        assert!(args.help);
        assert!(!args.version);
    }

    #[test]
    fn test_help_long() {
        let a = argv(&["--help"]);
        let args = parse_args_impl(&a);
        assert!(args.help);
    }

    #[test]
    fn test_version_short() {
        let a = argv(&["-V"]);
        let args = parse_args_impl(&a);
        assert!(args.version);
    }

    #[test]
    fn test_version_long() {
        let a = argv(&["--version"]);
        let args = parse_args_impl(&a);
        assert!(args.version);
    }

    #[test]
    fn test_list_models() {
        let a = argv(&["--list-models"]);
        let args = parse_args_impl(&a);
        assert!(args.list_models);
    }

    #[test]
    fn test_json() {
        let a = argv(&["--json", "img.png"]);
        let args = parse_args_impl(&a);
        assert!(args.json);
    }

    #[test]
    fn test_quiet() {
        let a = argv(&["--quiet", "img.png"]);
        let args = parse_args_impl(&a);
        assert!(args.quiet);
    }

    // ============================================================
    // 图片参数测试
    // ============================================================

    #[test]
    fn test_image_positional() {
        let a = argv(&["photo.jpg"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
    }

    #[test]
    fn test_image_with_path_flag() {
        let a = argv(&["--path", "photo.jpg"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
    }

    #[test]
    fn test_image_missing_no_flags() {
        let a = argv(&[]);
        let args = parse_args_impl(&a);
        assert!(args.image.as_os_str().is_empty());
        // help/version/list-models 均为 false 时，调用方应 exit(1)
    }

    #[test]
    fn test_image_missing_but_help() {
        let a = argv(&["--help"]);
        let args = parse_args_impl(&a);
        assert!(args.help);
        assert!(args.image.as_os_str().is_empty());
        // 有 help 标志时不应 exit
    }

    #[test]
    fn test_image_missing_but_version() {
        let a = argv(&["-V"]);
        let args = parse_args_impl(&a);
        assert!(args.version);
    }

    #[test]
    fn test_image_missing_but_list_models() {
        let a = argv(&["--list-models"]);
        let args = parse_args_impl(&a);
        assert!(args.list_models);
    }

    // ============================================================
    // -m / --model 测试
    // ============================================================

    #[test]
    fn test_model_v4() {
        let a = argv(&["img.png", "-m", "v4"]);
        let args = parse_args_impl(&a);
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/ch_PP-OCRv4_det_infer.mnn"))
        );
        assert_eq!(
            args.rec_model,
            Some(PathBuf::from("models/ch_PP-OCRv4_rec_infer.mnn"))
        );
        assert_eq!(
            args.keys,
            Some(PathBuf::from("models/ppocr_keys_v4.txt"))
        );
    }

    #[test]
    fn test_model_v5() {
        let a = argv(&["img.png", "-m", "v5"]);
        let args = parse_args_impl(&a);
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv5_mobile_det.mnn"))
        );
        assert_eq!(
            args.rec_model,
            Some(PathBuf::from("models/PP-OCRv5_mobile_rec.mnn"))
        );
        assert_eq!(
            args.keys,
            Some(PathBuf::from("models/ppocr_keys_v5.txt"))
        );
    }

    #[test]
    fn test_model_v6() {
        let a = argv(&["img.png", "-m", "v6"]);
        let args = parse_args_impl(&a);
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv6_small_det.mnn"))
        );
        assert_eq!(
            args.rec_model,
            Some(PathBuf::from("models/PP-OCRv6_small_rec.mnn"))
        );
        assert_eq!(
            args.keys,
            Some(PathBuf::from("models/ppocr_keys_v6.txt"))
        );
    }

    #[test]
    fn test_model_default_v5() {
        let a = argv(&["img.png"]);
        let args = parse_args_impl(&a);
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv5_mobile_det.mnn"))
        );
    }

    #[test]
    fn test_model_unknown_version() {
        let a = argv(&["img.png", "-m", "v99"]);
        let args = parse_args_impl(&a);
        // 未知版本回退到 v5
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv5_mobile_det.mnn"))
        );
    }

    #[test]
    fn test_m_json_alias() {
        let a = argv(&["img.png", "-m", "json"]);
        let args = parse_args_impl(&a);
        assert!(args.json);
        assert!(!args.quiet, "json 和 quiet 应保持独立");
        // 不改变模型路径
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv5_mobile_det.mnn"))
        );
    }

    #[test]
    fn test_json_flag_does_not_set_quiet() {
        let a = argv(&["img.png", "--json"]);
        let args = parse_args_impl(&a);
        assert!(args.json);
        assert!(!args.quiet, "--json 不应隐含 --quiet");
    }

    #[test]
    fn test_quiet_does_not_set_json() {
        let a = argv(&["img.png", "--quiet"]);
        let args = parse_args_impl(&a);
        assert!(args.quiet);
        assert!(!args.json, "--quiet 不应隐含 --json");
    }

    #[test]
    fn test_model_long_flag() {
        let a = argv(&["img.png", "--model", "v6"]);
        let args = parse_args_impl(&a);
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv6_small_det.mnn"))
        );
    }

    // ============================================================
    // 模型路径覆盖测试
    // ============================================================

    #[test]
    fn test_det_override() {
        let a = argv(&["img.png", "--det", "custom_det.mnn"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.det_model, Some(PathBuf::from("custom_det.mnn")));
    }

    #[test]
    fn test_rec_override() {
        let a = argv(&["img.png", "--rec", "custom_rec.mnn"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.rec_model, Some(PathBuf::from("custom_rec.mnn")));
    }

    #[test]
    fn test_keys_override() {
        let a = argv(&["img.png", "--keys", "custom_keys.txt"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.keys, Some(PathBuf::from("custom_keys.txt")));
    }

    #[test]
    fn test_ori_model() {
        let a = argv(&["img.png", "--ori", "models/ori.mnn"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.ori_model, Some(PathBuf::from("models/ori.mnn")));
    }

    // ============================================================
    // 其他选项测试
    // ============================================================

    #[test]
    fn test_output() {
        let a = argv(&["img.png", "--output", "out.png"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.output, Some(PathBuf::from("out.png")));
    }

    #[test]
    fn test_backend() {
        let a = argv(&["img.png", "--backend", "metal"]);
        let args = parse_args_impl(&a);
        assert!(matches!(args.backend, Backend::Metal));
    }

    #[test]
    fn test_threads() {
        let a = argv(&["img.png", "--threads", "8"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.threads, Some(8));
    }

    #[test]
    fn test_threads_invalid() {
        let a = argv(&["img.png", "--threads", "abc"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.threads, None);
    }

    // ============================================================
    // 组合测试
    // ============================================================

    #[test]
    fn test_path_json_compat() {
        // 外部系统调用格式: ocr-cli --path <img> -m json
        let a = argv(&["--path", "photo.jpg", "-m", "json"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
        assert!(args.json);
        assert!(!args.quiet, "外部调用不应隐含 --quiet");
    }

    #[test]
    fn test_path_m_json_combined() {
        // --path 和 -m json 可以任意顺序
        let a = argv(&["-m", "json", "--path", "photo.jpg"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
        assert!(args.json);
    }

    #[test]
    fn test_json_quiet_both() {
        let a = argv(&["img.png", "--json", "--quiet"]);
        let args = parse_args_impl(&a);
        assert!(args.json);
        assert!(args.quiet, "同时指定 --json --quiet 时 quiet 应为 true");
    }

    #[test]
    fn test_image_m_v6_json_output() {
        let a = argv(&["photo.jpg", "-m", "v6", "--json", "--output", "out.png"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv6_small_det.mnn"))
        );
        assert!(args.json);
        assert_eq!(args.output, Some(PathBuf::from("out.png")));
    }

    #[test]
    fn test_full_pipeline_args() {
        let a = argv(&[
            "scan.png",
            "-m", "v6",
            "--ori", "models/ori.mnn",
            "--backend", "cpu",
            "--threads", "4",
            "--json",
            "--output", "annotated.png",
        ]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("scan.png"));
        assert_eq!(
            args.det_model,
            Some(PathBuf::from("models/PP-OCRv6_small_det.mnn"))
        );
        assert_eq!(args.ori_model, Some(PathBuf::from("models/ori.mnn")));
        assert!(matches!(args.backend, Backend::CPU));
        assert_eq!(args.threads, Some(4));
        assert!(args.json);
        assert_eq!(args.output, Some(PathBuf::from("annotated.png")));
    }

    #[test]
    fn test_positional_after_flags() {
        let a = argv(&["--json", "photo.jpg"]);
        let args = parse_args_impl(&a);
        assert!(args.json);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
    }

    #[test]
    fn test_image_before_flags() {
        let a = argv(&["photo.jpg", "--json", "--quiet"]);
        let args = parse_args_impl(&a);
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
        assert!(args.json);
        assert!(args.quiet);
    }

    // ============================================================
    // output_text / output_json 测试
    // ============================================================

    fn make_result(text: &str, conf: f32, left: i32, top: i32, w: u32, h: u32) -> ocr_rs::OcrResult_ {
        use imageproc::rect::Rect;
        use ocr_rs::TextBox;
        let rect = Rect::at(left, top).of_size(w, h);
        let bbox = TextBox::new(rect, conf);
        ocr_rs::OcrResult_::new(text.to_string(), conf, bbox)
    }

    #[test]
    fn test_output_text_empty() {
        let results: Vec<ocr_rs::OcrResult_> = vec![];
        output_text(&results, true);
        output_text(&results, false);
    }

    #[test]
    fn test_output_text_single() {
        let results = vec![make_result("Hello", 0.95, 10, 20, 100, 30)];
        output_text(&results, true);
        output_text(&results, false);
    }

    #[test]
    fn test_output_text_multiple() {
        let results = vec![
            make_result("Line 1", 0.95, 10, 20, 100, 30),
            make_result("Line 2", 0.88, 10, 60, 120, 30),
            make_result("Line 3", 0.72, 10, 100, 90, 30),
        ];
        output_text(&results, true);
        output_text(&results, false);
    }

    #[test]
    fn test_output_json_empty() {
        let results: Vec<ocr_rs::OcrResult_> = vec![];
        let r = output_json(&results, false);
        assert!(r.is_ok());
    }

    #[test]
    fn test_output_json_single() {
        let results = vec![make_result("Hello", 0.95, 10, 20, 100, 30)];
        let r = output_json(&results, false);
        assert!(r.is_ok());
    }

    #[test]
    fn test_output_json_multiple() {
        let results = vec![
            make_result("Hello", 0.95, 10, 20, 100, 30),
            make_result("World", 0.88, 10, 60, 120, 30),
        ];
        let r = output_json(&results, false);
        assert!(r.is_ok());
    }

    #[test]
    fn test_output_json_quiet() {
        let results = vec![make_result("Test", 0.90, 0, 0, 50, 20)];
        let r = output_json(&results, true);
        assert!(r.is_ok());
    }

    // ============================================================
    // OcrResult_ 测试
    // ============================================================

    #[test]
    fn test_ocr_result_new() {
        use imageproc::rect::Rect;
        use ocr_rs::TextBox;
        let rect = Rect::at(5, 10).of_size(100, 30);
        let bbox = TextBox::new(rect, 0.95);
        let r = ocr_rs::OcrResult_::new("test".into(), 0.95, bbox);
        assert_eq!(r.text, "test");
        assert!((r.confidence - 0.95).abs() < f32::EPSILON);
        assert_eq!(r.bbox.rect.left(), 5);
        assert_eq!(r.bbox.rect.top(), 10);
        assert_eq!(r.bbox.rect.width(), 100);
        assert_eq!(r.bbox.rect.height(), 30);
    }
}

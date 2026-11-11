"""Package retained experiment artifacts and current source using only stdlib.

Run after report generation. Only the maintained reports are packaged.
Original input files remain in the project's Data directory.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (PROJECT_ROOT / "result").resolve()
DATASETS = ("PingData", "PlanetLab", "TaxiBJ_in", "TaxiBJ_out", "FT-AED", "UCSD-Ped2")
RESULT_DIRECTORIES = (
    "evaluation", "evaluation_H2_32", "paper_comparison", "paper_comparison_H2_32",
)
REPORT_FILES = (
    "QTJoint_六数据集_实验测试评估报告.docx",
    "QTJoint_实验结果展示.html",
    "QTJoint_H2_32_六数据集_实验测试评估报告.docx",
    "QTJoint_H2_32_实验结果展示.html",
    "QTJoint_vs_TKQT_六数据集_论文算法对比评估报告.docx",
    "QTJoint_vs_TKQT_对比结果.html",
)
SKIP_DIRECTORIES = {
    "__pycache__", "source_snapshot", "source_original", "tmp", "temp",
    ".tmp", ".temp", ".git", ".pytest_cache",
}
SKIP_FILES = {"delivery_verification.json", "final_delivery_check.json"}
ARCHIVE_NAME = "QTJoint_六数据集_实验数据与分析.zip"
NOTE_NAME = "project/result/归档说明.txt"


def within(path, parent):
    resolved = Path(path).resolve()
    if resolved != parent and parent not in resolved.parents:
        raise ValueError("Path is outside the permitted directory: " + str(resolved))
    return resolved


def excluded(path):
    parts = path.relative_to(PROJECT_ROOT).parts
    if any(part in SKIP_DIRECTORIES for part in parts):
        return True
    if any(part.lower().startswith((".tmp", "_tmp", "tmp_", "temp_")) for part in parts[:-1]):
        return True
    return (
        path.suffix.lower() in {".zip", ".pyc", ".pyo", ".tmp"}
        or path.name.lower().endswith(".sha256")
        or path.name.startswith("~$")
        or path.name in SKIP_FILES
    )


def collect_files():
    files = {}

    def add(path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("Symbolic links are not packaged: " + str(path))
        resolved = within(path, PROJECT_ROOT)
        if not resolved.is_file() or excluded(resolved):
            return
        relative = resolved.relative_to(PROJECT_ROOT)
        name = "project/" + relative.as_posix()
        if name in files:
            raise ValueError("Duplicate source entry: " + name)
        files[name] = resolved

    def add_directory(path):
        directory = within(path, PROJECT_ROOT)
        if not directory.is_dir():
            raise FileNotFoundError(str(directory))
        for current, directories, names in os.walk(str(directory), followlinks=False):
            current = Path(current)
            directories[:] = sorted(
                name for name in directories
                if not excluded(current / name) and not (current / name).is_symlink()
            )
            for name in sorted(names):
                add(current / name)

    for name in RESULT_DIRECTORIES:
        add_directory(RESULT_ROOT / name)
    for name in REPORT_FILES:
        path = RESULT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError("Generate the report before packaging: " + str(path))
        add(path)
    for path in sorted(RESULT_ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".log", ".txt"}:
            add(path)
    for path in sorted(PROJECT_ROOT.iterdir()):
        if path.is_file() and (path.suffix.lower() in {".py", ".md"} or path.name == "requirements.txt"):
            add(path)
    for name in ("experiments", "tests", "assets"):
        add_directory(PROJECT_ROOT / name)
    if (PROJECT_ROOT / "docs").is_dir():
        add_directory(PROJECT_ROOT / "docs")
    return sorted(files.items())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_results():
    within(RESULT_ROOT, PROJECT_ROOT)
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(str(RESULT_ROOT))
    output = within(RESULT_ROOT / ARCHIVE_NAME, RESULT_ROOT)
    temporary = within(RESULT_ROOT / ("." + output.stem + ".tmp.zip"), RESULT_ROOT)
    verification_path = within(RESULT_ROOT / "delivery_verification.json", RESULT_ROOT)
    checksum_path = within(RESULT_ROOT / (ARCHIVE_NAME + ".sha256"), RESULT_ROOT)
    if temporary.exists():
        raise FileExistsError("Temporary archive already exists: " + str(temporary))
    files = collect_files()
    note = (
        "QT-Joint 六数据集实验数据与分析\n\n"
        "数据集：" + "、".join(DATASETS) + "。\n"
        "本归档包含四套保留实验结果、三份 Word 报告及 HTML 展示，以及当前源码、测试和项目文档。\n"
        "源码位于 project/，结果位于 project/result/；experiments、tests 和 assets 保持在源码目录内的相对结构。\n"
        "原始数据仍位于项目 Data 目录；本归档不包含原始数据和 third_party 工具。\n"
        "结果目录中的 inputs 保存实验使用的标准输入数组，可结合元数据复核实验。\n"
        "项目目录结构和各报告中的复现说明应结合阅读。\n"
        "最终归档的 CRC 检查、SHA-256 和文件数记录在归档外的 delivery_verification.json 及 .zip.sha256 文件。\n"
    ).encode("utf-8")
    if NOTE_NAME in {name for name, _ in files}:
        raise ValueError("The generated archive note conflicts with a source entry")
    started = time.monotonic()
    try:
        with zipfile.ZipFile(str(temporary), "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for index, (name, path) in enumerate(files, 1):
                before = path.stat()
                archive.write(str(path), arcname=name)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError("Source changed during packaging: " + str(path))
                if index % 500 == 0:
                    print("Packaged {}/{} source files".format(index, len(files)), flush=True)
            archive.writestr(NOTE_NAME, note)

        with zipfile.ZipFile(str(temporary)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("Duplicate archive entries")
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise RuntimeError("Archive CRC failed: " + bad_entry)
            expected = {name for name, _ in files} | {NOTE_NAME}
            if set(names) != expected:
                raise RuntimeError("Archive entry set differs from source selection")
            if any(name.lower().endswith(".zip") for name in names):
                raise RuntimeError("Nested archives are not permitted")
            if archive.read(NOTE_NAME) != note:
                raise RuntimeError("Archive note does not match")

        # Both paths were resolved and confined to RESULT_ROOT before creation.
        os.replace(str(temporary), str(output))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    digest = sha256(output)
    verification = {
        "archive": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": digest,
        "archive_entry_count": len(files) + 1,
        "source_file_count": len(files),
        "datasets": list(DATASETS),
        "crc_passed": True,
        "duplicate_entries": 0,
        "nested_archives": 0,
        "reports": list(REPORT_FILES),
        "original_inputs": "Data (excluded from this archive)",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "verification_location": "Outside the archive to avoid self-referential checksums",
    }
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=True, indent=2), flush=True)
    return verification


if __name__ == "__main__":
    package_results()

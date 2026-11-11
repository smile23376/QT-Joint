"""Rebuild six-dataset reports from stored measurements without rerunning codecs.

Run from any directory with Python 3.10+ and Pillow.  --output-root supports a
preview directory; the input measurements under --result-root are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont

from result_report_document import Report


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("PingData", "PlanetLab", "TaxiBJ_in", "TaxiBJ_out", "FT-AED", "UCSD-Ped2")
BUDGETS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4, 0.5)
OLD_BITS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
ORDER = ("QT-Joint H2-32", "QT-Joint old H", "QT-Joint", "TKQT-independent", "SZ3", "ZFP", "SPERR", "MGARD-CPU")
COLORS = ("#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#353d4a", "#825b29")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    source = load_json(path)
    if not isinstance(source, list):
        raise ValueError(f"Expected a record list: {path}")
    return [r for r in source if r.get("dataset") in DATASETS]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def number(value, digits=4):
    return f"{value:.{digits}g}" if finite(value) else "—"


def percentage(value):
    return f"{100 * value:.4g}%" if finite(value) else "—"


def methods(rows):
    found = {r["method"] for r in rows}
    return [m for m in ORDER if m in found] + sorted(found.difference(ORDER))


def valid(rows):
    return [r for r in rows if finite(r.get("relative_error")) and r["relative_error"] >= 0
            and finite(r.get("cr")) and r["cr"] > 0]


def select_budget(rows, budget):
    candidates = [r for r in valid(rows) if r["relative_error"] <= budget]
    return max(candidates, key=lambda r: (r["cr"], -r["relative_error"], str(r.get("run_id", "")))) if candidates else None


def budget_rows(rows):
    result = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            for method in methods(rows):
                selected = select_budget([r for r in rows if r["dataset"] == dataset and r["method"] == method], budget)
                item = {"dataset": dataset, "budget": budget, "method": method, "covered": selected is not None}
                if selected:
                    for key in ("run_id", "relative_error", "cr", "file_bytes", "bpv", "ranks", "bits"):
                        item[key] = selected.get(key)
                result.append(item)
    return result


def audit_stored_budgets(recomputed, stored):
    lookup = {(r["dataset"], r["budget"], r["method"]): r for r in stored}
    mismatches = []
    for item in recomputed:
        key = item["dataset"], item["budget"], item["method"]
        reference = lookup.get(key)
        if reference is None or item["covered"] != reference.get("covered"):
            mismatches.append(key)
        elif item["covered"] and not math.isclose(item["cr"], reference["cr"], rel_tol=1e-10):
            mismatches.append(key)
    return {"compared": len(recomputed), "mismatch_count": len(mismatches), "mismatches": mismatches}


def font(size):
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/arial.ttf"),
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


class Canvas:
    """Draw the same simple chart as a PNG and an editable vector SVG."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']

    def line(self, coords, color="#d8e1e8", width=1):
        points = list(zip(coords[::2], coords[1::2]))
        self.draw.line(points, fill=color, width=width)
        xy = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def text(self, x, y, value, size=18, color="#203345", anchor="left"):
        value = str(value)
        face = font(size)
        bbox = self.draw.textbbox((0, 0), value, font=face)
        width = bbox[2] - bbox[0]
        start = x - width / 2 if anchor == "center" else x - width if anchor == "right" else x
        self.draw.text((start, y), value, font=face, fill=color)
        alignment = {"left": "start", "center": "middle", "right": "end"}[anchor]
        self.svg.append(f'<text x="{x:.2f}" y="{y + size:.2f}" text-anchor="{alignment}" font-family="Arial,Microsoft YaHei,sans-serif" font-size="{size}" fill="{color}">{html.escape(value)}</text>')

    def circle(self, x, y, radius, color):
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def rect(self, x0, y0, x1, y1, color):
        self.draw.rectangle((x0, y0, x1, y1), fill=color)
        self.svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" fill="{color}"/>')

    def save(self, stem):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(stem.with_suffix(".png"))
        stem.with_suffix(".svg").write_text("\n".join(self.svg) + "\n</svg>\n", encoding="utf-8")


def frontier(rows):
    result, best = [], -math.inf
    for row in sorted(valid(rows), key=lambda r: (r["relative_error"], -r["cr"])):
        if row["relative_error"] > 0 and row["cr"] > best:
            result.append(row)
            best = row["cr"]
    return result


def rate_distortion(rows, path, title):
    canvas = Canvas(1540, 1320)
    canvas.text(44, 20, title, size=30)
    palette = {method: COLORS[i % len(COLORS)] for i, method in enumerate(methods(rows))}
    for i, (method, color) in enumerate(palette.items()):
        x, y = 48 + (i % 4) * 375, 76 + (i // 4) * 32
        canvas.line((x, y+13, x+30, y+13), color, 3)
        canvas.text(x+40, y, method, size=18)
    for index, dataset in enumerate(DATASETS):
        x0, y0 = 104 + (index % 2) * 760, 202 + (index // 2) * 360
        w, h = 640, 240
        selected = valid([r for r in rows if r["dataset"] == dataset and r["relative_error"] > 0])
        canvas.text(x0+w/2, y0-44, dataset, size=24, anchor="center")
        if not selected:
            canvas.text(x0, y0, "No stored measurements", size=20)
            continue
        lx = [math.log10(r["relative_error"] * 100) for r in selected]
        ly = [math.log10(r["cr"]) for r in selected]
        xmin, xmax = math.floor(min(lx)), math.ceil(max(lx))
        ymin, ymax = math.floor(min(ly)), math.ceil(max(ly))
        if xmin == xmax: xmax += 1
        if ymin == ymax: ymax += 1
        def position(row):
            return (x0+w*(math.log10(row["relative_error"]*100)-xmin)/(xmax-xmin),
                    y0+h-h*(math.log10(row["cr"])-ymin)/(ymax-ymin))
        for exponent in range(xmin, xmax+1, max(1, math.ceil((xmax-xmin)/5))):
            x = x0+w*(exponent-xmin)/(xmax-xmin)
            canvas.line((x, y0, x, y0+h))
            canvas.text(x, y0+h+8, f"1e{exponent}", size=15, anchor="center")
        for exponent in range(ymin, ymax+1):
            y = y0+h-h*(exponent-ymin)/(ymax-ymin)
            canvas.line((x0, y, x0+w, y))
            canvas.text(x0-12, y-12, f"{10**exponent:g}", size=15, anchor="right")
        canvas.line((x0, y0, x0, y0+h, x0+w, y0+h), "#203345", 2)
        canvas.text(x0, y0-24, "actual CR", size=15)
        canvas.text(x0+w/2, y0+h+38, "relative error (%) - logarithmic axes", size=17, anchor="center")
        for method, color in palette.items():
            points = frontier([r for r in selected if r["method"] == method])
            xy = [value for row in points for value in position(row)]
            if len(points) > 1: canvas.line(xy, color, 3)
            for row in points:
                x, y = position(row)
                canvas.circle(x, y, 3, color)
    canvas.text(48, 1280, "Stored discrete Pareto points; connecting lines are visual guides, not interpolated measurements.", size=17)
    canvas.save(path)


def coverage_figure(budgets, path):
    palette = {method: COLORS[i % len(COLORS)] for i, method in enumerate(methods(budgets))}
    height = 160 + 70 * len(palette)
    canvas = Canvas(1400, height)
    canvas.text(38, 24, "Budget coverage across six retained datasets", size=30)
    canvas.text(420, 76, "Covered dataset-budget pairs / 48", size=20)
    canvas.text(1120, 76, "At 10% / 6", size=20)
    for i, (method, color) in enumerate(palette.items()):
        y = 126 + 70*i
        selected = [r for r in budgets if r["method"] == method]
        covered = sum(r["covered"] for r in selected)
        at10 = sum(r["covered"] for r in selected if r["budget"] == 0.1)
        canvas.text(38, y, method, size=21)
        canvas.rect(420, y+4, 1000, y+32, "#edf2f6")
        if covered: canvas.rect(420, y+4, 420+580*covered/48, y+32, color)
        canvas.text(1020, y, f"{covered}/48", size=21)
        canvas.text(1180, y, f"{at10}/6", size=21)
    canvas.save(path)


def common_intro(report, title, scope):
    report.p(title, "Title")
    report.p("六数据集保留版 · 2026 年 11 月 11 日")
    report.p("本报告由既有实测记录筛选后重新汇总生成。本次维护没有重新运行压缩算法；重新计算的内容为记录计数、离散率失真前沿和误差预算内的最优实际压缩率。")
    report.p(scope)
    report.p("误差使用原始数值域的相对 Frobenius 范数 ||X-X̂||F/||X||F。实际压缩率为原始 float32 数组的 4N 字节除以可解码载荷字节数；QT-Joint 包含模型与归一化侧文件。理想模型位数不能代替实际文件大小。")


def data_section(report, manifest, main, all_rows):
    report.h("1 数据规模与记录范围")
    table = []
    for dataset in DATASETS:
        info = manifest[dataset]
        shape = info["shape"]
        count = math.prod(shape)
        table.append([dataset, "×".join(map(str, shape)), f"{count:,}", f"{count*4/2**20:.2f}",
                      sum(r["dataset"] == dataset for r in main), sum(r["dataset"] == dataset for r in all_rows)])
    report.table(["数据集", "完整评测形状", "元素数", "FP32 MiB", "主比较记录", "源配置记录"], table, small=True)
    report.p("源配置记录一列在旧 H 报告中对应 all_records，在 H2-32 报告中对应 all_new_records（不含复用参考），在论文对比报告中对应 comparison_records。它与主比较记录存在重合，不能相加。")
    report.p("表中完整输入尺度不代表所有附加实验的输入尺度；附加实验可使用子张量或独立窗口，不能直接与完整输入合并比较。")
    report.p("UCSD-Ped2 使用 2010 个测试帧缩放为 24×36，属于该分辨率下的张量压缩；FT-AED 的原始 Frobenius 误差汇总混合量纲通道。PingData 轴语义及部分数据物理单位尚无完整来源确认。")


def metric_section(report, main, all_rows, figures):
    report.h("2 误差、压缩率与离散率失真")
    summary = []
    for method in methods(main):
        selected = valid([r for r in main if r["method"] == method])
        summary.append([method, len(selected),
                        percentage(min((r["relative_error"] for r in selected), default=None)),
                        percentage(max((r["relative_error"] for r in selected), default=None)),
                        number(min((r["cr"] for r in selected), default=None)),
                        number(max((r["cr"] for r in selected), default=None))])
    report.table(["方法", "记录数", "最小误差", "最大误差", "最小实际CR", "最大实际CR"], summary, small=True)
    report.p("上表是各方法已测试配置的范围，最小误差和最大压缩率通常来自不同运行，不能组合成一个运行结果。参数网格的配置数量也不相同。")
    report.figure("retained_rate_distortion", "图 1 六数据集的离散 Pareto 前沿。仅保留误差更小或压缩率更高的非支配点；连线仅辅助阅读。")
    auxiliary = Counter(r["method"] for r in all_rows)
    report.p("本套源配置记录的方法分布：" + "；".join(f"{m} {auxiliary[m]} 条" for m in sorted(auxiliary)) + "。")


def budget_section(report, main, budgets):
    report.h("3 误差预算覆盖与可比配置")
    report.p("对 1%、2%、5%、10%、15%、20%、40%、50% 八个预算，分别在每个数据集、每种方法的既有离散网格中选择满足误差不超过预算且实际压缩率最大的运行。未覆盖表示当前记录中没有满足预算的配置，不表示方法理论上无法达到该误差；不插值，也不外推。")
    report.figure("retained_budget_coverage", "图 2 预算覆盖数量。共 6×8=48 个数据集与预算组合；右列单独显示 10% 预算覆盖。")
    columns = methods(main)
    detail = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            cells = []
            for method in columns:
                item = next(r for r in budgets if r["dataset"] == dataset and r["budget"] == budget and r["method"] == method)
                cells.append(number(item.get("cr")) + "×" if item["covered"] else "未覆盖")
            detail.append([dataset, percentage(budget)] + cells)
    report.table(["数据集", "预算"] + columns, detail, small=True)
    report.p("每格为满足该预算的最高实际压缩率；完整选中运行编号及其实际误差保存于本套 retained_report_summary.json 的 budget_results 中。")
    leaders = []
    for dataset in DATASETS:
        eligible = [r for r in budgets if r["dataset"] == dataset and r["budget"] == 0.1 and r["covered"]]
        best = max(eligible, key=lambda r: r["cr"]) if eligible else None
        leaders.append([dataset, best["method"] if best else "未覆盖", percentage(best.get("relative_error")) if best else "—", number(best.get("cr")) if best else "—"])
    report.table(["数据集", "10%预算内当前网格最高CR方法", "该运行实际误差", "实际CR"], leaders)
    report.p("上述预算内选择只支持当前输入、实现和已测试参数网格下的比较。不同误差处的最高压缩率不能直接比较，预算覆盖不足的数据集也不能用其余数据集的均值代替。")


def pair_section(report, pairs, main):
    ids = {r["run_id"] for r in main if r["method"] == "QT-Joint H2-32"}
    selected = [r for r in pairs if r.get("method") == "QT-Joint" and r.get("run_id") in ids]
    report.h("4 位宽集合扩大后的成对变化")
    report.p(f"完整输入主比较中保留 {len(selected)} 对相同运行编号的旧、新记录。只汇总主比较，不将子张量、时窗和消融的目标值混在一起。")
    table = []
    for dataset in DATASETS:
        rows = [r for r in selected if r["dataset"] == dataset]
        comparable = [r for r in rows if r.get("ranks_same") and r.get("delta_same")]
        savings = [r["file_saving"] for r in comparable if finite(r.get("file_saving"))]
        table.append([dataset, len(rows), len(comparable),
                      sum(r["new_error"] < r["old_error"] for r in comparable),
                      sum(r["new_file_bytes"] < r["old_file_bytes"] for r in comparable),
                      percentage(statistics.median(savings)) if savings else "—"])
    report.table(["数据集", "配对数", "同秩同权重", "误差降低数", "实际文件减小数", "文件节省中位数"], table, small=True)
    report.p("文件节省=(旧字节数−新字节数)/旧字节数；负值表示文件增大。扩大位宽候选集合不保证实际 NPZ 文件更小，也不保证有限迭代下每个运行的误差严格降低。32 bit 整数量化仍不同于无量化 float32。")
    return selected


def robustness_section(report, rows):
    report.h("4 缺失输入补充结果")
    if not rows:
        report.p("当前保留记录中没有缺失输入补充结果。")
        return
    table = []
    for row in rows:
        metrics = row.get("metrics", {})
        table.append([row["dataset"], row["method"], row.get("configuration", "—"),
                      percentage(metrics.get("all", {}).get("nrmse")),
                      percentage(metrics.get("missing", {}).get("nrmse")), number(row.get("cr_full"))])
    report.table(["数据集", "方法", "配置", "完整真值误差", "缺失处误差", "完整数组口径CR"], table, small=True)
    report.p("该补充实验仅涉及 PlanetLab 的前 6 个时间片、一次缺失掩码及其记录配置，不能推广为六数据集的鲁棒性结论。这里的误差与完整真值比较；cr_full 的分子包含缺失位置，不能与按观测值字节数计算的 cr_observed 混用。补全后的 QT 分支属于外接处理，独立于 QT-Joint 核心压缩目标。")


def limitations(report, paths):
    report.h("事实限制与数据文件")
    report.p("所有数值来自保留的既有实测记录。本次只维护结果集合并重算汇总，没有新增压缩实测，没有重新测量运行时间或内存，也不据此作效率排序。")
    report.p("TKQT-independent 是按论文描述编写的独立实现，不是作者代码；其结果依赖本地秩、量化和编码设置，不能直接视为原论文数值的复现。QT-Joint 采用有限候选秩与有限迭代，目标下降或坐标扫描状态不等于联合全局最优证明。")
    report.p("来源文件：" + "；".join(paths) + "。完整逐运行数据应结合这些 JSON/CSV 及保存的模型检查。")


def build_suite(result_root, output_root, suite, manifest):
    if suite == "evaluation":
        all_rows = records(result_root / suite / "all_records.json")
        main = records(result_root / suite / "main_records.json")
        title, word_name, html_name = "QT-Joint 六数据集实验测试评估", "QTJoint_六数据集_实验测试评估报告.docx", "QTJoint_实验结果展示.html"
        scope = "本套保留旧位宽集合 H=(2,3,4,5,6,7,8,10,12,16) 的 QT-Joint 记录，比较基线为 SZ3、ZFP、SPERR 和 MGARD-CPU。这里的旧 H 结果与 H={2,…,32} 的结果分别报告。"
        paths = ["evaluation/all_records.json", "evaluation/main_records.json", "evaluation/dataset_manifest.json"]
    elif suite == "evaluation_H2_32":
        all_rows = records(result_root / suite / "all_new_records.json")
        main = records(result_root / suite / "combined_main_records.json")
        title, word_name, html_name = "QT-Joint H={2,…,32} 六数据集实验测试评估", "QTJoint_H2_32_六数据集_实验测试评估报告.docx", "QTJoint_H2_32_实验结果展示.html"
        scope = "本套 QT-Joint H2-32 使用 H={2,3,…,32} 共 31 个整数位宽。主比较同时列出旧 H、TKQT-independent 与外部基线的既有参考记录，并明确方法标签；这些参考记录没有在本次维护中重新运行。"
        paths = ["evaluation_H2_32/all_new_records.json", "evaluation_H2_32/combined_main_records.json", "evaluation_H2_32/paired_old_new.json", "evaluation_H2_32/budget_results.json"]
    else:
        all_rows = main = records(result_root / suite / "comparison_records.json")
        title, word_name, html_name = "QT-Joint 与 TKQT 独立实现六数据集对比评估", "QTJoint_vs_TKQT_六数据集_论文算法对比评估报告.docx", "QTJoint_vs_TKQT_对比结果.html"
        scope = "本套比较旧 H=(2,3,4,5,6,7,8,10,12,16) 的 QT-Joint 与 TKQT-independent。后者为论文算法独立实现，包含其本地秩和量化配置网格；不是作者发布代码，也不将本地数值等同于原论文结果。"
        paths = ["paper_comparison/comparison_records.json", "paper_comparison/robustness_results.json"]
    figures = output_root / suite / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    budgets = budget_rows(main)
    budget_audit = None
    if suite == "evaluation_H2_32":
        budget_audit = audit_stored_budgets(budgets, records(result_root / suite / "budget_results.json"))
        if budget_audit["mismatch_count"]:
            raise ValueError("Stored budget results differ from the retained main-record recomputation: " + str(budget_audit))
    rate_distortion(main, figures / "retained_rate_distortion", suite + " | six retained datasets")
    coverage_figure(budgets, figures / "retained_budget_coverage")
    report = Report(figure_dir=figures)
    common_intro(report, title, scope)
    data_section(report, manifest, main, all_rows)
    metric_section(report, main, all_rows, figures)
    budget_section(report, main, budgets)
    pairs = []
    robustness = []
    if suite == "evaluation_H2_32":
        pairs = pair_section(report, records(result_root / suite / "paired_old_new.json"), main)
    elif suite == "paper_comparison":
        robustness = records(result_root / suite / "robustness_results.json")
        robustness_section(report, robustness)
    limitations(report, paths)
    summary = {"datasets": list(DATASETS), "rebuilt_from_stored_measurements": True,
               "compression_algorithms_rerun": False, "suite": suite,
               "main_record_count": len(main), "suite_record_count": len(all_rows),
               "method_counts": dict(Counter(r["method"] for r in main)),
               "budget_results": budgets, "main_pair_count": len(pairs),
               "robustness_record_count": len(robustness), "stored_budget_audit": budget_audit,
               "source_files": paths}
    summary_path = output_root / suite / "retained_report_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.save(output_root / word_name, html_path=output_root / html_name)
    return {"suite": suite, "main_records": len(main), "suite_records": len(all_rows),
            "word": str(output_root / word_name), "html": str(output_root / html_name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "result")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--suites", nargs="+", choices=("evaluation", "evaluation_H2_32", "paper_comparison"),
                        default=["evaluation", "evaluation_H2_32", "paper_comparison"])
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    output_root = (args.output_root or result_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_source = load_json(result_root / "evaluation" / "dataset_manifest.json")
    manifest = {name: manifest_source[name] for name in DATASETS}
    output = [build_suite(result_root, output_root, suite, manifest) for suite in args.suites]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

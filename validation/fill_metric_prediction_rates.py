"""One-off helper to fill prediction anomaly rates in docs/metric.md."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC_PATH = ROOT / "docs" / "metric.md"

MODELS = {
    "v2": ROOT / "submission",
    "v3": ROOT / "submission_v3",
    "v4": ROOT / "submission_v4_NoRegime",
}


def anomaly_rate(csv_path: Path) -> tuple[int, int, float]:
    df = pd.read_csv(csv_path)
    if "y_pred" not in df.columns:
        raise ValueError(f"{csv_path} does not contain y_pred")
    positives = int(df["y_pred"].sum())
    total = int(len(df))
    return positives, total, positives / total


def replace_rates_for_section(lines: list[str], header_index: int, simple_text: str, complex_text: str) -> None:
    task_lines = []
    for index in range(header_index + 1, len(lines)):
        stripped = lines[index].lstrip()
        if stripped.startswith("### "):
            break
        if "task1" in lines[index].lower():
            task_lines.append(index)
        elif "task2" in lines[index].lower():
            task_lines.append(index)

    if len(task_lines) < 2:
        raise RuntimeError(f"Could not find task1/task2 placeholders after line {header_index + 1}")

    task1_prefix = lines[task_lines[0]].split("：", 1)[0]
    task2_prefix = lines[task_lines[1]].split("：", 1)[0]
    lines[task_lines[0]] = f"{task1_prefix}：{simple_text}\n"
    lines[task_lines[1]] = f"{task2_prefix}：{complex_text}\n"


def main() -> None:
    results = {}
    for model, directory in MODELS.items():
        simple_pos, simple_total, simple_rate = anomaly_rate(directory / "pred_simple.csv")
        complex_pos, complex_total, complex_rate = anomaly_rate(directory / "pred_complex.csv")
        results[model] = {
            "simple": f"{simple_rate * 100:.2f}% ({simple_pos}/{simple_total})",
            "complex": f"{complex_rate * 100:.2f}% ({complex_pos}/{complex_total})",
        }

    lines = METRIC_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    headers = {}
    for index, line in enumerate(lines):
        lowered = line.lower()
        if line.startswith("### ") and "v2" in lowered:
            headers["v2"] = index
        elif line.startswith("### ") and "v3" in lowered:
            headers["v3"] = index
        elif line.startswith("### ") and "v4" in lowered:
            headers["v4"] = index

    missing = sorted(set(MODELS) - set(headers))
    if missing:
        raise RuntimeError(f"Missing metric sections: {missing}")

    for model in ["v2", "v3", "v4"]:
        replace_rates_for_section(
            lines,
            headers[model],
            results[model]["simple"],
            results[model]["complex"],
        )

    METRIC_PATH.write_text("".join(lines), encoding="utf-8")

    for model, values in results.items():
        print(f"{model}: task1={values['simple']} task2={values['complex']}")


if __name__ == "__main__":
    main()

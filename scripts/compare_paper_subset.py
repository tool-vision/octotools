#!/usr/bin/env python3
"""Compare a scored two-task subset against paper-reported OctoTools results."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


PAPER_RESULTS = {
    "gameof24": {
        "paper_name": "Game of 24",
        "octotools": 44.7,
        "std": 2.8,
        "zero_shot": 22.2,
        "cot": 33.3,
    },
    "gpqa": {
        "paper_name": "GPQA",
        "octotools": 54.7,
        "std": 1.3,
        "zero_shot": 53.7,
        "cot": 52.3,
    },
}


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError(f"Unsafe expression node: {type(node).__name__}")


def numbers_in_expression(expression: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", expression)]


def expression_candidates(text: str) -> list[str]:
    normalized = (
        text.replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace(r"\[", " ")
        .replace(r"\]", " ")
        .replace(r"\(", " ")
        .replace(r"\)", " ")
    )
    candidates = []

    for match in re.findall(r"`([^`]+)`", normalized):
        candidates.append(match)

    math_chars = r"[0-9\s+\-*/().]+"
    for match in re.findall(math_chars, normalized):
        candidate = match.strip()
        if len(candidate) >= 3 and any(op in candidate for op in "+-*/"):
            candidates.append(candidate)

    return candidates


def score_gameof24(response: str, expected_numbers: list[int]) -> tuple[bool, str]:
    expected_counter = Counter(expected_numbers)
    for candidate in expression_candidates(response):
        if Counter(numbers_in_expression(candidate)) != expected_counter:
            continue
        try:
            parsed = ast.parse(candidate, mode="eval")
            value = safe_eval(parsed)
        except Exception:
            continue
        if abs(value - 24.0) < 1e-6:
            return True, candidate
    return False, ""


def extract_choice(response: str) -> str | None:
    patterns = [
        r"(?i)\banswer\s*[:\-]?\s*\(?([A-D])\)?\b",
        r"(?i)\boption\s+([A-D])\b",
        r"(?i)\bchoice\s+([A-D])\b",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, response))
    if matches:
        return matches[-1].upper()

    standalone = re.findall(r"\b([A-D])\b", response.upper())
    return standalone[-1] if standalone else None


def score_gpqa(response: str, answer: str) -> tuple[bool, str]:
    extracted = extract_choice(response)
    return extracted == answer.upper(), extracted or ""


def result_file(run_dir: Path, task: str, index: int) -> Path:
    return run_dir / "results" / task / f"output_{index}.json"


def selected_result_paths(run_dir: Path, task: str) -> list[Path]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return sorted((run_dir / "results" / task).glob("output_*.json"))

    rows = load_json(summary_path)
    paths = []
    for row in rows:
        if row.get("task") != task:
            continue
        if row.get("status") not in {"passed", "skipped"}:
            continue
        if not row.get("has_direct_output"):
            continue
        paths.append(Path(row["output_file"]))
    return sorted(paths, key=lambda path: int(path.stem.replace("output_", "")))


def confidence_interval(correct: int, total: int) -> dict[str, float | None]:
    if total == 0:
        return {"ci95_low": None, "ci95_high": None}
    p = correct / total
    se = math.sqrt(p * (1 - p) / total)
    return {
        "ci95_low": round(max(0.0, (p - 1.96 * se) * 100), 2),
        "ci95_high": round(min(100.0, (p + 1.96 * se) * 100), 2),
    }


def score_task(run_dir: Path, task: str) -> dict[str, Any]:
    data = load_json(REPO_ROOT / "tasks" / task / "data" / "data.json")
    result_paths = selected_result_paths(run_dir, task)
    rows = []

    for path in result_paths:
        index = int(path.stem.replace("output_", ""))
        result = load_json(path)
        datum = data[index]
        response = result.get("direct_output", "")
        if task == "gameof24":
            correct, extracted = score_gameof24(response, datum["question"])
        elif task == "gpqa":
            correct, extracted = score_gpqa(response, datum["answer"])
        else:
            raise ValueError(f"Unsupported task for deterministic scoring: {task}")
        rows.append(
            {
                "task": task,
                "index": index,
                "pid": result.get("pid"),
                "correct": correct,
                "expected": datum["answer"],
                "extracted": extracted,
                "step_count": result.get("step_count"),
                "output_file": str(path),
            }
        )

    correct_count = sum(1 for row in rows if row["correct"])
    total = len(rows)
    accuracy = round(correct_count / total * 100, 2) if total else 0.0
    paper = PAPER_RESULTS[task]
    ci = confidence_interval(correct_count, total)
    return {
        "task": task,
        "paper_name": paper["paper_name"],
        "sample_correct": correct_count,
        "sample_total": total,
        "sample_accuracy": accuracy,
        **ci,
        "paper_reference_accuracy": paper["octotools"],
        "paper_std": paper["std"],
        "subset_delta_unadjusted": round(accuracy - paper["octotools"], 2),
        "paper_zero_shot": paper["zero_shot"],
        "paper_cot": paper["cot"],
        "rows": rows,
    }


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "paper_name",
        "sample_correct",
        "sample_total",
        "sample_accuracy",
        "ci95_low",
        "ci95_high",
        "paper_reference_accuracy",
        "paper_std",
        "subset_delta_unadjusted",
        "paper_zero_shot",
        "paper_cot",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: summary[name] for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score gameof24/gpqa subset and compare with paper Table 1."
    )
    parser.add_argument("run_dir", help="Run directory containing results/<task>/output_*.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    preset = None
    if manifest_path.exists():
        preset = load_json(manifest_path).get("preset")

    summaries = []
    for task in ["gameof24", "gpqa"]:
        if (run_dir / "results" / task).exists():
            summaries.append(score_task(run_dir, task))

    if preset == "paper-two-task-500":
        bad = [summary for summary in summaries if summary["sample_total"] != 20]
        if bad:
            details = ", ".join(
                f"{summary['task']}={summary['sample_total']}" for summary in bad
            )
            raise SystemExit(
                "paper-two-task-500 comparison requires exactly 20 scored "
                f"outputs per task; got {details}"
            )

    if not summaries:
        raise SystemExit(f"No supported task results found under {run_dir}")

    output_json = run_dir / "paper_comparison.json"
    output_csv = run_dir / "paper_comparison.csv"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    write_csv(output_csv, summaries)

    for summary in summaries:
        print(
            f"{summary['paper_name']}: "
            f"subset {summary['sample_accuracy']}% "
            f"({summary['sample_correct']}/{summary['sample_total']}), "
            f"paper reference {summary['paper_reference_accuracy']} "
            f"+/- {summary['paper_std']}, "
            f"unadjusted delta {summary['subset_delta_unadjusted']:+.2f}"
        )
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print("Note: subset accuracy is not statistically comparable to the full paper run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

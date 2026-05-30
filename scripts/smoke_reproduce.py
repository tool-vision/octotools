#!/usr/bin/env python3
"""Offline smoke test for the tiny OctoTools reproduction harness."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "reproduce_octotools.py"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="octotools-smoke-") as temp_dir:
        output_root = Path(temp_dir)
        run_dir = run_dry_run(output_root, "tiny-demo", "smoke-local")
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        summary_csv = run_dir / "summary.csv"

        assert_true(manifest_path.exists(), "manifest.json was not created")
        assert_true(summary_path.exists(), "summary.json was not created")
        assert_true(summary_csv.exists(), "summary.csv was not created")

        manifest = load_json(manifest_path)
        summary = load_json(summary_path)

        assert_true(manifest["dry_run"] is True, "manifest did not record dry_run")
        assert_true(manifest["engine"] == "gpt-4o-mini", "unexpected default engine")
        assert_true(len(manifest["jobs"]) == 5, "tiny-demo should define 5 jobs")
        assert_true(len(summary) == 5, "summary should contain 5 jobs")

        for job in manifest["jobs"]:
            command_text = " ".join(job["command"])
            assert_true(job["data_file"].endswith("data/data.json"), "bad data file")
            assert_true(Path(job["data_file"]).exists(), "data file missing")
            assert_true(job["tools"], "job has no tools")
            assert_true(job["command"][0] == sys.executable, "command must use sys.executable")
            assert_true("tasks/solve.py" in command_text, "command must call tasks/solve.py")
            assert_true("--max_steps 2" in command_text, "dry run should cap max steps")

        for row in summary:
            assert_true(row["status"] == "dry-run", "summary ran a real job")
            assert_true(not Path(row["output_file"]).exists(), "dry run wrote task output")

        paper_run_dir = run_dry_run(
            output_root, "paper-two-task-500", "smoke-paper-two-task"
        )
        paper_manifest = load_json(paper_run_dir / "manifest.json")
        paper_summary = load_json(paper_run_dir / "summary.json")
        assert_true(len(paper_manifest["jobs"]) == 40, "paper preset should define 40 jobs")
        assert_true(len(paper_summary) == 40, "paper preset summary should contain 40 jobs")
        assert_true(
            paper_manifest["request_budget"]["estimated_openai_requests"] == 400,
            "paper preset should estimate 400 requests at max_steps=2",
        )
        task_indices = {}
        for job in paper_manifest["jobs"]:
            task_indices.setdefault(job["task"], []).append(job["index"])
            forbidden_tools = {
                "Google_Search_Tool",
                "Advanced_Object_Detector_Tool",
                "Pubmed_Search_Tool",
                "Text_Detector_Tool",
            }
            assert_true(
                not forbidden_tools.intersection(job["tools"]),
                "paper preset should not require non-OpenAI tools",
            )
        assert_true(task_indices["gameof24"] == list(range(100, 120)), "bad gameof24 indices")
        assert_true(task_indices["gpqa"] == list(range(100, 120)), "bad gpqa indices")

    print("smoke_reproduce.py passed")
    return 0


def run_dry_run(output_root: Path, preset: str, label: str) -> Path:
    command = [
        sys.executable,
        str(RUNNER),
        "--preset",
        preset,
        "--dry-run",
        "--label",
        label,
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert_true(
        result.returncode == 0,
        f"{preset} dry run failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    return output_root / preset / label


if __name__ == "__main__":
    raise SystemExit(main())

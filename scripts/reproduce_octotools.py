#!/usr/bin/env python3
"""Run small, reproducible OctoTools benchmark presets.

The default preset is intentionally tiny. It validates the paper-style task
execution pipeline without trying to reproduce the full published tables.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"
OCTOTOOLS_DIR = REPO_ROOT / "octotools"


PRESETS: dict[str, list[dict[str, Any]]] = {
    "tiny-demo": [
        {
            "task": "gameof24",
            "indices": [0, 1],
            "tools": [
                "Python_Code_Generator_Tool",
                "Generalist_Solution_Generator_Tool",
            ],
        },
        {
            "task": "gpqa",
            "indices": [0],
            "tools": ["Generalist_Solution_Generator_Tool"],
        },
        {
            "task": "medqa",
            "indices": [0],
            "tools": ["Generalist_Solution_Generator_Tool"],
        },
        {
            "task": "mmlu-pro",
            "indices": [0],
            "tools": [
                "Wikipedia_Knowledge_Searcher_Tool",
                "Generalist_Solution_Generator_Tool",
            ],
        },
    ]
    ,
    "paper-two-task-500": [
        {
            "task": "gameof24",
            "indices": list(range(100, 120)),
            "tools": [
                "Python_Code_Generator_Tool",
                "Generalist_Solution_Generator_Tool",
            ],
        },
        {
            "task": "gpqa",
            "indices": list(range(100, 120)),
            "tools": ["Generalist_Solution_Generator_Tool"],
        },
    ],
}


LLM_BACKED_TOOLS = {
    "Generalist_Solution_Generator_Tool",
    "Image_Captioner_Tool",
    "Python_Code_Generator_Tool",
    "Relevant_Patch_Zoomer_Tool",
}


def load_dotenv_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE entries from a .env file without external deps."""

    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        from dotenv import dotenv_values

        return {
            key: value
            for key, value in dotenv_values(path).items()
            if key and value is not None
        }
    except Exception:
        pass

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def merged_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in load_dotenv_file(REPO_ROOT / ".env").items():
        env.setdefault(key, value)
    return env


def parse_indices(raw: str | None) -> dict[str, list[int]]:
    """Parse task:index,index specs into a task-to-indices mapping."""

    if not raw:
        return {}

    parsed: dict[str, list[int]] = {}
    for chunk in raw.split(";"):
        if not chunk.strip():
            continue
        task, _, values = chunk.partition(":")
        if not task or not values:
            raise ValueError(
                "--indices must use 'task:0,1;other-task:2' format"
            )
        parsed[task.strip()] = [int(value) for value in values.split(",") if value]
    return parsed


def data_file_for(task: str) -> Path:
    return TASKS_DIR / task / "data" / "data.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def env_status(engine: str, env: dict[str, str] | None = None) -> dict[str, bool]:
    env = env if env is not None else os.environ
    required = []
    if engine.startswith(("gpt", "o1", "o3", "o4")):
        required.append("OPENAI_API_KEY")
    if engine.startswith("forge/"):
        required.append("FORGE_API_KEY")
    if "claude" in engine:
        required.append("ANTHROPIC_API_KEY")
    if "gemini" in engine:
        required.append("GOOGLE_API_KEY")
    if "deepseek" in engine:
        required.append("DEEPSEEK_API_KEY")
    if "grok" in engine:
        required.append("XAI_API_KEY")
    if "together" in engine:
        required.append("TOGETHER_API_KEY")
    return {name: bool(env.get(name)) for name in required}


def estimated_requests_for_job(job: dict[str, Any], max_steps: int) -> int:
    """Conservative OpenAI-call estimate for one solver job.

    For output_types=direct, the benchmark solver makes one query-analysis call,
    one direct-output call, and for each step at least planning, command, and
    verification calls. If the selected tool is LLM-backed, add one more call per
    step. We estimate using the worst LLM-backed tool in the job's tool list.
    """

    per_step = 3
    if any(tool in LLM_BACKED_TOOLS for tool in job["tools"]):
        per_step += 1
    return 2 + (max_steps * per_step)


def estimate_request_budget(jobs: list[dict[str, Any]], max_steps: int) -> dict[str, int]:
    per_job = [estimated_requests_for_job(job, max_steps) for job in jobs]
    return {
        "jobs": len(jobs),
        "max_steps": max_steps,
        "estimated_openai_requests": sum(per_job),
        "max_estimated_requests_per_job": max(per_job) if per_job else 0,
    }


def validate_config(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = []
    for entry in entries:
        task = entry["task"]
        data_file = data_file_for(task)
        if not data_file.exists():
            raise FileNotFoundError(f"Missing data file for task '{task}': {data_file}")
        data = load_json(data_file)
        total = len(data)
        for index in entry["indices"]:
            if index < 0 or index >= total:
                raise IndexError(
                    f"Index {index} is invalid for task '{task}' with {total} rows"
                )
        if not entry["tools"]:
            raise ValueError(f"Task '{task}' must declare at least one tool")
        validated.append({**entry, "data_count": total})
    return validated


def build_jobs(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    jobs = []
    for entry in entries:
        task = entry["task"]
        data_file = data_file_for(task)
        result_dir = run_dir / "results" / task
        cache_dir = run_dir / "cache" / task
        log_dir = run_dir / "logs" / task
        tools = ",".join(entry["tools"])

        for index in entry["indices"]:
            output_file = result_dir / f"output_{index}.json"
            command = [
                sys.executable,
                str(TASKS_DIR / "solve.py"),
                "--index",
                str(index),
                "--task",
                task,
                "--data_file",
                str(data_file),
                "--llm_engine_name",
                args.engine,
                "--root_cache_dir",
                str(cache_dir),
                "--output_json_dir",
                str(result_dir),
                "--output_types",
                args.output_types,
                "--enabled_tools",
                tools,
                "--max_time",
                str(args.max_time),
                "--max_steps",
                str(args.max_steps),
            ]
            if args.max_tokens is not None:
                command.extend(["--max_tokens", str(args.max_tokens)])
            jobs.append(
                {
                    "task": task,
                    "index": index,
                    "tools": entry["tools"],
                    "command": command,
                    "output_file": str(output_file),
                    "log_file": str(log_dir / f"{index}.log"),
                    "result_dir": str(result_dir),
                    "cache_dir": str(cache_dir),
                    "data_file": str(data_file),
                    "max_steps": args.max_steps,
                }
            )
    return jobs


def read_output_summary(output_file: str) -> dict[str, Any]:
    path = Path(output_file)
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "pid": data.get("pid"),
        "step_count": data.get("step_count"),
        "execution_time": data.get("execution_time"),
        "has_direct_output": bool(data.get("direct_output")),
        "has_final_output": bool(data.get("final_output")),
        "has_base_response": bool(data.get("base_response")),
    }


def is_complete_output(output_file: str, max_steps: int) -> bool:
    summary = read_output_summary(output_file)
    step_count = summary.get("step_count")
    return bool(summary.get("has_direct_output")) and (
        step_count is None or step_count <= max_steps
    )


def run_job(job: dict[str, Any], env: dict[str, str], skip_existing: bool) -> dict[str, Any]:
    output_file = Path(job["output_file"])
    log_file = Path(job["log_file"])

    if skip_existing and output_file.exists() and is_complete_output(
        job["output_file"], job["max_steps"]
    ):
        summary = read_output_summary(job["output_file"])
        return {**job, **summary, "status": "skipped", "returncode": 0, "seconds": 0.0}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    Path(job["cache_dir"]).mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            job["command"],
            cwd=OCTOTOOLS_DIR,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = round(time.monotonic() - start, 2)
    status = "passed" if process.returncode == 0 and output_file.exists() else "failed"
    summary = read_output_summary(job["output_file"])
    return {
        **job,
        **summary,
        "status": status,
        "returncode": process.returncode,
        "seconds": seconds,
    }


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    budget: dict[str, int],
    env: dict[str, str] | None = None,
) -> Path:
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "preset": args.preset,
        "engine": args.engine,
        "output_types": args.output_types,
        "max_steps": args.max_steps,
        "max_time": args.max_time,
        "max_tokens": args.max_tokens,
        "dry_run": args.dry_run,
        "skip_existing": args.skip_existing,
        "env": env_status(args.engine, env),
        "request_budget": budget,
        "tasks": entries,
        "jobs": jobs,
    }
    path = run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = [
        "task",
        "index",
        "status",
        "returncode",
        "seconds",
        "pid",
        "step_count",
        "execution_time",
        "has_direct_output",
        "output_file",
        "log_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})

    return json_path, csv_path


def make_run_dir(output_root: Path, preset: str, label: str | None) -> Path:
    safe_label = label or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / preset / safe_label


def build_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset '{args.preset}'. Available: {sorted(PRESETS)}")
    overrides = parse_indices(args.indices)
    entries = []
    for entry in PRESETS[args.preset]:
        if overrides and entry["task"] not in overrides:
            continue
        indices = overrides.get(entry["task"], entry["indices"])
        entries.append({**entry, "indices": indices})
    if not entries:
        raise ValueError("No jobs selected. Check --preset and --indices.")
    return validate_config(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small reproducible OctoTools benchmark preset."
    )
    parser.add_argument("--preset", default="tiny-demo", choices=sorted(PRESETS))
    parser.add_argument("--engine", default="gpt-4o-mini")
    parser.add_argument(
        "--indices",
        help="Optional override, e.g. 'gameof24:0,1;gpqa:0'.",
    )
    parser.add_argument("--label", help="Run label under the output root.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--output-types", default="direct")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--request-budget",
        type=int,
        default=500,
        help="Maximum estimated OpenAI requests allowed for a non-dry run.",
    )
    parser.add_argument(
        "--reserve-requests",
        type=int,
        default=100,
        help="Requests reserved for retries/manual follow-up within --request-budget.",
    )
    parser.add_argument(
        "--allow-over-budget",
        action="store_true",
        help="Run even when the conservative request estimate exceeds --request-budget.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = merged_environment()
    run_dir = make_run_dir(Path(args.output_root).resolve(), args.preset, args.label)
    entries = build_entries(args)
    jobs = build_jobs(entries, args, run_dir)
    budget = estimate_request_budget(jobs, args.max_steps)
    manifest_path = write_manifest(run_dir, args, entries, jobs, budget, env)

    missing_env = [
        name for name, is_set in env_status(args.engine, env).items() if not is_set
    ]
    if missing_env and not args.dry_run:
        names = ", ".join(missing_env)
        print(f"Missing required environment variable(s): {names}", file=sys.stderr)
        print(f"Manifest written to: {manifest_path}", file=sys.stderr)
        return 2

    estimated_requests = budget["estimated_openai_requests"]
    allowed_requests = args.request_budget - args.reserve_requests
    if (
        estimated_requests > allowed_requests
        and not args.allow_over_budget
        and not args.dry_run
    ):
        print(
            "Estimated OpenAI requests exceed budget: "
            f"{estimated_requests} > {allowed_requests} "
            f"({args.request_budget} budget - {args.reserve_requests} reserve)",
            file=sys.stderr,
        )
        print("Use --allow-over-budget to run anyway.", file=sys.stderr)
        print(f"Manifest written to: {manifest_path}", file=sys.stderr)
        return 3

    if args.dry_run:
        results = [
            {**job, "status": "dry-run", "returncode": None, "seconds": 0.0}
            for job in jobs
        ]
        summary_json, summary_csv = write_summary(run_dir, results)
        print(f"Dry run complete: {len(jobs)} jobs")
        print(
            "Estimated OpenAI requests: "
            f"{estimated_requests} "
            f"(budget {args.request_budget}, reserve {args.reserve_requests})"
        )
        print(f"Manifest: {manifest_path}")
        print(f"Summary JSON: {summary_json}")
        print(f"Summary CSV: {summary_csv}")
        return 0

    python_path = os.pathsep.join(
        [str(REPO_ROOT), str(OCTOTOOLS_DIR), env.get("PYTHONPATH", "")]
    )
    env["PYTHONPATH"] = python_path

    workers = max(1, args.threads)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(run_job, job, env, args.skip_existing): job for job in jobs
        }
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(
                f"{result['status']}: {result['task']}[{result['index']}] "
                f"returncode={result['returncode']} seconds={result['seconds']}"
            )

    results.sort(key=lambda item: (item["task"], item["index"]))
    summary_json, summary_csv = write_summary(run_dir, results)
    failed = [result for result in results if result["status"] == "failed"]
    print(f"Manifest: {manifest_path}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batch worker: run the OctoTools solver over a manifest of instances.

Reads a JSONL manifest where each line is:
    {"id": "<unique id>", "question": "<query text>", "image": "/abs/path.jpg" | null}

Each worker process builds the solver once (planner + executor + toolbox) and
solves its instances sequentially with a fresh Memory per instance, mirroring
tasks/solve.py (paper setting: max_steps=10, max_time=300, output_types=direct).
Results are written one JSON per instance under <output-root>/results/<id>.json;
existing results are skipped so reruns resume.

Requires an OpenAI-compatible server for the LLM engine, e.g.:
    export VLLM_BASE_URL=http://localhost:8222/v1
    python scripts/batch_worker.py --manifest m.jsonl --output-root out/ \
        --llm-engine vllm-Qwen/Qwen3-VL-8B-Instruct --num-workers 32
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_worker_solver = None
_worker_args = None


def _init_worker(args_dict):
    global _worker_solver, _worker_args
    _worker_args = args_dict
    from octotools.solver import construct_solver

    _worker_solver = construct_solver(
        llm_engine_name=args_dict["llm_engine"],
        enabled_tools=args_dict["enabled_tools"],
        output_types=args_dict["output_types"],
        max_steps=args_dict["max_steps"],
        max_time=args_dict["max_time"],
        max_tokens=args_dict["max_tokens"],
        root_cache_dir=os.path.join(args_dict["output_root"], "solver_cache"),
        verbose=False,
    )


def solve_instance(instance):
    from octotools.models.memory import Memory

    instance_id = str(instance["id"])
    output_root = _worker_args["output_root"]
    result_path = os.path.join(output_root, "results", f"{instance_id}.json")

    result = {"id": instance_id, "answer": None, "error": None}
    try:
        _worker_solver.memory = Memory()
        _worker_solver.root_cache_dir = os.path.join(
            output_root, "solver_cache", instance_id
        )
        image_path = instance.get("image") or None
        json_data = _worker_solver.solve(instance["question"], image_path)
        result["answer"] = json_data.get("direct_output") or json_data.get(
            "final_output"
        )
        trajectory_path = os.path.join(
            output_root, "trajectories", f"{instance_id}.json"
        )
        with open(trajectory_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
    except Exception:
        result["error"] = traceback.format_exc()

    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, result_path)
    status = "ok" if result["answer"] else "no-answer"
    print(f"[{status}] {instance_id}", flush=True)
    return instance_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--llm-engine", default="vllm-Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--enabled-tools",
        default="Generalist_Solution_Generator_Tool",
        help="comma-separated tool list (paper OctoTools_base default)",
    )
    parser.add_argument("--output-types", default="direct")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    instances = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    results_dir = os.path.join(args.output_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "trajectories"), exist_ok=True)
    pending = [
        inst
        for inst in instances
        if not os.path.exists(os.path.join(results_dir, f"{inst['id']}.json"))
    ]
    print(
        f"{len(instances)} instances, {len(instances) - len(pending)} done, {len(pending)} to run",
        flush=True,
    )
    if not pending:
        print("BATCH_COMPLETE", flush=True)
        return

    # Fail fast if the LLM endpoint is down: a broken pool initializer would
    # otherwise respawn workers forever. (Checked only when work remains, so
    # fully-finished manifests don't require a live server.)
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8888/v1")
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            r.read()
    except Exception as exc:
        print(f"FATAL: LLM endpoint {base_url} unreachable: {exc}", flush=True)
        sys.exit(2)

    args_dict = {
        "llm_engine": args.llm_engine,
        "enabled_tools": args.enabled_tools.split(","),
        "output_types": args.output_types,
        "max_steps": args.max_steps,
        "max_time": args.max_time,
        "max_tokens": args.max_tokens,
        "output_root": args.output_root,
    }
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=min(args.num_workers, len(pending)),
        initializer=_init_worker,
        initargs=(args_dict,),
    ) as pool:
        for _ in pool.imap_unordered(solve_instance, pending):
            pass
    print("BATCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

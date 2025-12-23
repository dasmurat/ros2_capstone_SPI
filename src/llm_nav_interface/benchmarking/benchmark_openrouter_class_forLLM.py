#!/usr/bin/env python3
"""
Benchmark multiple LLMs via OpenRouter with warmup + trials.
- Creates a fresh timestamped results directory per run inside the provided results path
- Saves both CSV and JSONL with rich metadata

Usage example:
python3 benchmark_openrouter.py \
  --models ~/turtlebot4_ws/src/llm_nav_interface/models.yaml \
  --warmup 1 \
  --trials 1 \
  --commands "go to the docking station" "I'm thirsty, get me water" \
  --output-csv ~/turtlebot4_ws/src/llm_nav_interface/results/results.csv \
  --raw-jsonl ~/turtlebot4_ws/src/llm_nav_interface/results/results_raw.jsonl

Environment:
  OPENROUTER_API_KEY   (required)

Notes:
- We intentionally ignore the exact filenames in --output-csv / --raw-jsonl and
  instead use their directory to create a timestamped subfolder like
  results/results_YYYY-MM-DD_HH-MM-SS/{results.csv, results_raw.jsonl}
- If a model call fails, we still log a row with error info.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    print("[fatal] PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --------------------------- Utilities --------------------------- #

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------- Data Models --------------------------- #

@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: str
    model: str
    command: str
    trial_index: int
    is_warmup: int
    latency_sec: Optional[float]
    tokens_prompt: Optional[int]
    tokens_completion: Optional[int]
    tokens_total: Optional[int]
    parsed_ok: int
    error: Optional[str]
    response_text: Optional[str]

    def to_csv_row(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------- Benchmark Runner --------------------------- #

class BenchmarkRunner:
    def __init__(
        self,
        models_config_path: str,
        commands: List[str],
        warmup: int,
        trials: int,
        output_csv_arg: str,
        raw_jsonl_arg: str,
        request_timeout: int = 120,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.models_config_path = os.path.expanduser(models_config_path)
        self.commands = commands
        self.warmup = max(0, int(warmup))
        self.trials = max(1, int(trials))
        self.request_timeout = request_timeout
        self.system_prompt = system_prompt or (
            "You are a helpful navigation assistant. Interpret the user's natural language "
            "command for a mobile robot and respond succinctly."
        )

        # Prepare a timestamped results directory under the directory of the provided paths
        self.run_id = run_timestamp()
        base_results_dir = os.path.dirname(os.path.expanduser(output_csv_arg))
        if not base_results_dir:
            base_results_dir = "."
        self.run_dir = os.path.join(base_results_dir, f"results_{self.run_id}")
        ensure_dir(self.run_dir)

        # Final file paths inside the run_dir
        self.csv_path = os.path.join(self.run_dir, "results.csv")
        self.jsonl_path = os.path.join(self.run_dir, "results_raw.jsonl")

        # Load API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print("[fatal] OPENROUTER_API_KEY is not set in environment.", file=sys.stderr)
            sys.exit(1)

        # Load model list from YAML
        self.models = self._load_models_yaml(self.models_config_path)
        if not self.models:
            print("[fatal] No models found in YAML.", file=sys.stderr)
            sys.exit(1)

        # Prepare CSV with header
        self._init_csv()
    # filepath: /home/muratpc/turtlebot4_ws/src/llm_nav_interface/llm_nav_interface/benchmark_openrouter_class.py
    # ...existing code...
    @staticmethod
    def _load_models_yaml(path: str) -> list[str]:
        def extract_ids(node):
            found = []
            if isinstance(node, dict):
                # Common cases:
                if "models" in node:
                    found += extract_ids(node["models"])
                if "id" in node and isinstance(node["id"], str):
                    found.append(node["id"])
                for v in node.values():
                    found += extract_ids(v)
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, str):
                        found.append(it)
                    else:
                        found += extract_ids(it)
            elif isinstance(node, str):
                found.append(node)
            return found

        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        print(f"[debug] YAML docs loaded: {docs}")  # <--- Add this line

        models = []
        for doc in docs:
            if doc is not None:
                models += extract_ids(doc)

        # Dedup, preserve order
        seen, uniq = set(), []
        for m in (str(x).strip() for x in models):
            if m and m not in seen:
                seen.add(m); uniq.append(m)

        print(f"[debug] Extracted model IDs: {uniq}")  # <--- Add this line

        if not uniq:
            print(f"[fatal] No models found in YAML. Read {len(docs)} document(s). "
                f"Top-level types: {[type(d).__name__ if d is not None else None for d in docs]}", file=sys.stderr)
        return uniq
    # ...existing code...

    # ---- CSV/JSONL helpers ---- #
    def _init_csv(self) -> None:
        header = list(BenchmarkResult(
            run_id=self.run_id,
            timestamp=now_iso(),
            model="",
            command="",
            trial_index=0,
            is_warmup=0,
            latency_sec=None,
            tokens_prompt=None,
            tokens_completion=None,
            tokens_total=None,
            parsed_ok=0,
            error=None,
            response_text=None,
        ).to_csv_row().keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    def _append_csv(self, result: BenchmarkResult) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.to_csv_row().keys())
            writer.writerow(result.to_csv_row())

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- API call ---- #
    def _query_openrouter(self, model: str, prompt: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            # Keep responses short to reduce cost/latency; adjust if you need more text
            "max_tokens": 256,
        }

        start = time.perf_counter()
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=self.request_timeout)
            latency = time.perf_counter() - start
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:500]}", latency
            return resp.json(), None, latency
        except Exception as e:
            latency = time.perf_counter() - start
            return None, str(e), latency

    # ---- Run one call and persist results ---- #
    def _run_once(self, model: str, command: str, trial_index: int, is_warmup: bool) -> None:
        payload, err, latency = self._query_openrouter(model, command)

        # Defaults
        tokens_prompt = tokens_completion = tokens_total = None
        response_text = None
        parsed_ok = 0

        if payload is not None:
            try:
                # OpenRouter responses are OpenAI-compatible
                choice = payload.get("choices", [{}])[0]
                response_text = (choice.get("message", {}) or {}).get("content")

                usage = payload.get("usage", {})
                tokens_prompt = usage.get("prompt_tokens")
                tokens_completion = usage.get("completion_tokens")
                tokens_total = usage.get("total_tokens")

                parsed_ok = 1 if isinstance(response_text, str) and len(response_text.strip()) > 0 else 0
            except Exception as e:
                err = f"parse_error: {e}"
                parsed_ok = 0

        result = BenchmarkResult(
            run_id=self.run_id,
            timestamp=now_iso(),
            model=model,
            command=command,
            trial_index=trial_index,
            is_warmup=int(is_warmup),
            latency_sec=round(latency, 3) if latency is not None else None,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            parsed_ok=parsed_ok,
            error=err,
            response_text=(response_text if not is_warmup else None),  # avoid storing warmup bodies to save space
        )

        # Write CSV
        self._append_csv(result)

        # Write JSONL (store full payload if available)
        raw_record = {
            "run_id": self.run_id,
            "timestamp": result.timestamp,
            "model": model,
            "command": command,
            "trial_index": trial_index,
            "is_warmup": int(is_warmup),
            "latency_sec": result.latency_sec,
            "error": err,
            "payload": payload,  # keep the whole raw response
        }
        self._append_jsonl(raw_record)

    # ---- Public API ---- #
    def run(self) -> Tuple[str, str, str]:
        print(f"[info] Run ID: {self.run_id}")
        print(f"[info] Results directory: {self.run_dir}")
        print(f"[info] Models: {', '.join(self.models)}")
        print(f"[info] Commands: {self.commands}")
        print(f"[info] Warmup per model: {self.warmup} | Trials per command: {self.trials}")

        # Warmup (per model). We warm up using the first command if present, else a simple ping
        warmup_prompt = self.commands[0] if self.commands else "hello"
        for model in self.models:
            for w in range(self.warmup):
                self._run_once(model=model, command=warmup_prompt, trial_index=w, is_warmup=True)

        # Trials
        for model in self.models:
            for t in range(self.trials):
                for cmd in self.commands:
                    self._run_once(model=model, command=cmd, trial_index=t, is_warmup=False)

        print(f"[ok] CSV saved to: {self.csv_path}")
        print(f"[ok] JSONL saved to: {self.jsonl_path}")
        return self.run_id, self.csv_path, self.jsonl_path


# --------------------------- CLI --------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LLMs via OpenRouter with warmups and trials.")
    p.add_argument("--models", required=True, help="Path to YAML file listing models (e.g., {models: [openai/gpt-4o, mistralai/mistral-large]})")
    p.add_argument("--warmup", type=int, default=1, help="Warmup requests per model (default: 1)")
    p.add_argument("--trials", type=int, default=1, help="Number of trials per command per model (default: 1)")
    p.add_argument("--commands", nargs="*", default=[], help="Commands to send (space-separated strings)")
    p.add_argument("--output-csv", required=True, help="Base path under which timestamped results directory is created; filename ignored except for parent dir")
    p.add_argument("--raw-jsonl", required=True, help="Base path under which timestamped results directory is created; filename ignored except for parent dir")
    p.add_argument("--timeout", type=int, default=120, help="HTTP request timeout seconds (default: 120)")
    p.add_argument("--system-prompt", default=None, help="Optional system prompt override")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    runner = BenchmarkRunner(
        models_config_path=args.models,
        commands=args.commands,
        warmup=args.warmup,
        trials=args.trials,
        output_csv_arg=args.output_csv,
        raw_jsonl_arg=args.raw_jsonl,
        request_timeout=args.timeout,
        system_prompt=args.system_prompt,
    )
    run_id, csv_path, jsonl_path = runner.run()
    print(f"[summary] run_id={run_id}\n         csv={csv_path}\n       jsonl={jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

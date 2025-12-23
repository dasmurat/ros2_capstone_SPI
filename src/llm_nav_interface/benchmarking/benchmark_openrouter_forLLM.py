#!/usr/bin/env python3
"""
Benchmark multiple LLMs via OpenRouter with warmup + trials.
- Creates a fresh timestamped results directory per run inside ./results
- Saves CSV (and optional JSONL) with raw responses, latency, token usage, parse flag
- Flattens grouped models from models.yaml

Usage example:
python3 benchmark_openrouter.py \
  --models ./models.yaml \
  --warmup 1 \
  --trials 1 \
  --commands "go to the docking station" "I'm thirsty, get me water" \
  --collect-usage \
  --raw-jsonl ./results/latest_all_raw.jsonl

Environment:
  OPENROUTER_API_KEY   (required)
"""

import os
import csv
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests

# --- Optional dependency for YAML ---
try:
    import yaml
except Exception as e:
    yaml = None

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FIELDNAMES = [
    "timestamp", "model", "command", "latency_s", "parsed_ok",
    "tokens_in", "tokens_out", "tokens_total", "json", "raw"
]


# ---------- Utilities ----------
def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def make_results_dir(base: Path = Path("results")) -> Path:
    base.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / f"results_{ts}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def load_models_yaml(path: str) -> List[Dict[str, Any]]:
    """
    Load models.yaml that may contain grouped sections like:
      models:
        premium:
          - id: openai/gpt-5
            max_calls: 40
        open_baseline:
          - id: mistralai/mistral-7b-instruct
            max_calls: 30
    Returns a flat list: [{"id": "...", "max_calls": 40}, ...]
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read models.yaml. Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg or "models" not in cfg:
        raise ValueError("models.yaml must contain a top-level 'models' key.")

    models = []
    m = cfg["models"]
    # m can be a dict of groups, or already a flat list
    if isinstance(m, dict):
        for group, items in m.items():
            if not items:
                continue
            for it in items:
                if not isinstance(it, dict) or "id" not in it:
                    continue
                models.append({"id": it["id"], "max_calls": it.get("max_calls", None)})
    elif isinstance(m, list):
        for it in m:
            if not isinstance(it, dict) or "id" not in it:
                continue
            models.append({"id": it["id"], "max_calls": it.get("max_calls", None)})
    else:
        raise ValueError("Unsupported 'models' format in YAML.")

    if not models:
        raise ValueError("No models found in models.yaml.")
    return models


def extract_json_block(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from the text.
    Handles cases where model wraps output in ```json ... ``` or ``` ... ```
    Returns the JSON substring or None.
    """
    s = text.strip()
    # code fences
    if s.startswith("```"):
        # remove leading ```
        s = s[3:]
        # strip an optional 'json' language hint
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
        # remove trailing ```
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # Find the first '{' and the last '}' to grab a JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return None


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        maybe = extract_json_block(text) or text
        return json.loads(maybe)
    except Exception:
        return None


def validate_planner_schema(obj: Dict[str, Any]) -> bool:
    """
    Expecting something like:
    {"intent":"...", "goal":{"x": float, "y": float}, "note":"..."}
    We'll just check required keys and types lightly.
    """
    try:
        if not isinstance(obj, dict):
            return False
        if "intent" not in obj or "goal" not in obj:
            return False
        goal = obj["goal"]
        if not isinstance(goal, dict):
            return False
        # x,y can be strings or numbers; try casting
        for k in ("x", "y"):
            if k not in goal:
                return False
            float(goal[k])
        # note is optional
        return True
    except Exception:
        return False


# ---------- OpenRouter client ----------
class OpenRouterClient:
    def __init__(self, api_key: str, timeout: float = 60.0):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional routing headers:
            # "HTTP-Referer": "https://your-app-or-local",
            # "X-Title": "MDS650 Benchmark",
        }
        self.timeout = timeout

    def call(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Returns (text, usage_dict). usage_dict keys: prompt_tokens, completion_tokens, total_tokens
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = requests.post(
            OPENROUTER_URL, headers=self.headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenRouter response format (OpenAI-like)
        text = ""
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = json.dumps(data)  # fallback raw

        usage = data.get("usage", {}) or {}
        usage_map = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        return text, usage_map


# ---------- Core benchmark ----------
def run_single_test(
    client: OpenRouterClient,
    model_id: str,
    cmd: str,
    *,
    keep_raw: bool,
    max_raw_chars: int,
    collect_usage: bool,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    response_text, usage = client.call(
        model=model_id,
        system_prompt=system_prompt,
        user_prompt=cmd,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0

    parsed = safe_parse_json(response_text)
    ok = 1 if (parsed is not None and validate_planner_schema(parsed)) else 0

    parsed_json_str = json.dumps(parsed, ensure_ascii=False) if parsed else ""
    raw_for_csv = response_text[:max_raw_chars] if keep_raw else ""

    row = {
        "timestamp": now_iso(),
        "model": model_id,
        "command": cmd,
        "latency_s": round(latency, 3),
        "parsed_ok": ok,
        "tokens_in": usage.get("prompt_tokens", 0) if collect_usage else 0,
        "tokens_out": usage.get("completion_tokens", 0) if collect_usage else 0,
        "tokens_total": usage.get("total_tokens", 0) if collect_usage else 0,
        "json": parsed_json_str,
        "raw": raw_for_csv,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via OpenRouter")
    parser.add_argument("--models", type=str, required=True,
                        help="Path to models.yaml")
    parser.add_argument("--commands", nargs="+", required=True,
                        help="Commands to test (same list for all models)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup requests per command per model (not logged)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Benchmark trials per command per model (logged)")
    parser.add_argument("--collect-usage", action="store_true",
                        help="Record token usage if provided by API")
    parser.add_argument("--keep-raw", action="store_true", default=True,
                        help="Always store the raw text response in the CSV")
    parser.add_argument("--max-raw-chars", type=int, default=8000,
                        help="Truncate raw response to this many characters when saving")
    parser.add_argument("--raw-jsonl", type=str, default=None,
                        help="Optional path to also write a JSONL log of all responses")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Optional max_tokens for the completion")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="HTTP timeout per request (seconds)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt to enforce JSON format")
    parser.add_argument("--output-csv", type=str, default=None,
                    help="Optional path to write a CSV summary of all results")
    args = parser.parse_args()

    # Results directory for this run
    run_dir = make_results_dir(Path("results"))
    output_csv = run_dir / "results.csv"
    output_jsonl_path = Path(args.raw_jsonl) if args.raw_jsonl else None

    # Load models
    models = load_models_yaml(args.models)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    client = OpenRouterClient(api_key=api_key, timeout=args.timeout)

    commands = list(args.commands)
    M = len(models)
    C = len(commands)
    W = max(0, int(args.warmup))
    T = max(1, int(args.trials))

    # Pre-flight summary
    planned_logged = M * C * T
    planned_warmups = M * C * W
    print("\n=== Benchmark Plan ===")
    print(f"Models active      : {M}")
    print(f"Commands per model : {C}")
    print(f"Warmups per cmd    : {W}")
    print(f"Trials per cmd     : {T}")
    print(f"Total warmups      : {planned_warmups} (not logged)")
    print(f"Total logged calls : {planned_logged}")
    print(f"Results folder     : {run_dir}")
    print("======================\n")

    # Optional: write JSONL progressively
    jf = None
    if output_jsonl_path:
        # If a relative path was given, place it inside this run_dir for convenience
        if not output_jsonl_path.is_absolute():
            output_jsonl_path = run_dir / output_jsonl_path
        jf = open(output_jsonl_path, "w", encoding="utf-8")

    results: List[Dict[str, Any]] = []

    # Optional: default system prompt enforcing JSON
    default_system = args.system_prompt or (
        "You are a planner that converts a short natural-language instruction into a JSON object "
        "with fields: intent (string), goal {x: float, y: float}, note (string). "
        "Respond ONLY with valid compact JSON. If the command is unclear, return "
        '{"intent":"clarify","goal":{"x":0,"y":0},"note":"reason"}.'
    )

    # Loop
    for m in models:
        model_id = m["id"]
        # model cap is informational; not enforced unless you want to
        max_cap = m.get("max_calls", None)

        for cmd in commands:
            # Warmups (not logged)
            for _ in range(W):
                try:
                    _ = run_single_test(
                        client, model_id, cmd,
                        keep_raw=False,
                        max_raw_chars=args.max_raw_chars,
                        collect_usage=args.collect_usage,
                        system_prompt=default_system,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[warmup][{model_id}] error: {e}")

            # Trials (logged)
            for t in range(T):
                try:
                    row = run_single_test(
                        client, model_id, cmd,
                        keep_raw=args.keep_raw,
                        max_raw_chars=args.max_raw_chars,
                        collect_usage=args.collect_usage,
                        system_prompt=default_system,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    results.append(row)

                    if jf is not None:
                        full_row = dict(row)
                        # If you want *full* untruncated raw in JSONL, re-call without truncation:
                        # full_row["raw"] = full_row["raw"]  # here it's already truncated; adjust if needed
                        json.dump(full_row, jf, ensure_ascii=False)
                        jf.write("\n")

                    print(f"[ok] {model_id} | {cmd!r} | latency={row['latency_s']}s | parsed_ok={row['parsed_ok']}")
                except Exception as e:
                    print(f"[error][{model_id}] {cmd!r} -> {e}")
                    # still log a row with error context
                    results.append({
                        "timestamp": now_iso(),
                        "model": model_id,
                        "command": cmd,
                        "latency_s": -1.0,
                        "parsed_ok": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "tokens_total": 0,
                        "json": "",
                        "raw": f"ERROR: {e}",
                    })

    if jf is not None:
        jf.close()

    if args.output_csv:
        # put CSV next to JSONL if timestamped_dir isn't available here
        if args.raw_jsonl:
            run_dir = os.path.dirname(args.raw_jsonl)
        else:
            run_dir = "results"
            os.makedirs(run_dir, exist_ok=True)

        output_csv_path = os.path.join(run_dir, args.output_csv)
        fieldnames = list(results[0].keys())
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote CSV   : {output_csv_path}")

    if args.raw_jsonl:
        print(f"Wrote JSONL : {args.raw_jsonl}")

    print("\n=== Done ===")
    print("=======================")


if __name__ == "__main__":
    main()
# EOF
#!/usr/bin/env python3
"""下载 Databricks Dolly-15k，转成本仓库的 instruction/input/output JSON，并保留 category。

DATA-D1：不切 IID/Dirichlet（那是 DATA-D2）。默认 90/10 划一份共享 eval，seed 与闪卡相同。

用法::

    python scripts/prepare_dolly_15k.py
    python scripts/prepare_dolly_15k.py --out-dir data
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

HF_JSONL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
    "/resolve/main/databricks-dolly-15k.jsonl"
)
HF_MIRROR_JSONL = (
    "https://hf-mirror.com/datasets/databricks/databricks-dolly-15k"
    "/resolve/main/databricks-dolly-15k.jsonl"
)
SEED = 20260831


def convert_row(raw: dict) -> dict:
    return {
        "instruction": raw.get("instruction") or "",
        "input": raw.get("context") or "",
        "output": raw.get("response") or "",
        "category": raw.get("category") or "",
    }


def download_jsonl(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in (HF_JSONL, HF_MIRROR_JSONL):
        print(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "fedscale-dolly-prep/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            print(f"Wrote {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception as exc:  # noqa: BLE001 — 镜像失败再试下一个
            last_err = exc
            print(f"  failed: {exc}")
    raise SystemExit(f"Could not download Dolly-15k: {last_err}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--jsonl", type=Path, default=None, help="已有 jsonl 则跳过下载")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.jsonl or (out_dir / "databricks-dolly-15k.jsonl")
    if not jsonl_path.exists():
        download_jsonl(jsonl_path)
    else:
        print(f"Using existing {jsonl_path}")

    converted = [convert_row(r) for r in load_jsonl(jsonl_path)]
    cats: dict[str, int] = {}
    for r in converted:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    all_path = out_dir / "dolly_15k.json"
    all_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(converted)} rows -> {all_path}")
    print("category counts:")
    for k in sorted(cats):
        print(f"  {k}: {cats[k]}")

    rng = random.Random(args.seed)
    idx = list(range(len(converted)))
    rng.shuffle(idx)
    n_eval = int(round(len(idx) * args.eval_ratio))
    eval_rows = [converted[i] for i in idx[:n_eval]]
    train_rows = [converted[i] for i in idx[n_eval:]]
    train_path = out_dir / "dolly_15k_train.json"
    eval_path = out_dir / "dolly_15k_eval.json"
    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train={len(train_rows)} -> {train_path}")
    print(f"eval={len(eval_rows)} -> {eval_path} (seed={args.seed}, ratio={args.eval_ratio})")


if __name__ == "__main__":
    main()

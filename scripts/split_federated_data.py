#!/usr/bin/env python3
"""联邦数据切分（DATA-1）：支持均分 / 不等分 / Dirichlet 非 IID。

输出每 client 一份 train.json，并写 manifest。每机构 hold-out eval（DATA-2）。

用法::

    python scripts/split_federated_data.py \
        --train data/medical_flashcards_train.json \
        --eval data/medical_flashcards_eval.json \
        --out-dir data/splits \
        --num-clients 2 \
        --method dirichlet --alpha 0.5 --seed 20260831 \
        --holdout-eval-ratio 0.1

method:
  - uniform:   均匀切分（默认，等价当前 50/50）
  - fixed:     --ratios 0.7,0.3 不等分
  - dirichlet: 按 label/来源做 Dirichlet 非 IID（alpha 越小越 non-IID）
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_of(row: Dict[str, Any]) -> str:
    """用 input 首词或 output 首词作为弱 label（medical flashcards 无显式 label）。"""
    return (row.get("input", "") or row.get("output", ""))[:40]


def split_uniform(rows: List, n: int) -> List[List]:
    idx = list(range(len(rows)))
    random.shuffle(idx)
    chunks = [[] for _ in range(n)]
    for i, r in enumerate(idx):
        chunks[i % n].append(rows[r])
    return chunks


def split_fixed(rows: List, ratios: List[float]) -> List[List]:
    idx = list(range(len(rows)))
    random.shuffle(idx)
    out: List[List] = []
    start = 0
    total = sum(ratios)
    for r in ratios:
        end = start + int(round(len(idx) * r / total))
        out.append([rows[i] for i in idx[start:end]])
        start = end
    return out


def split_dirichlet(rows: List, n: int, alpha: float, by_label_fn) -> List[List]:
    """按 label 做 Dirichlet 非 IID：每个 label 下，各 client 份额 ~ Dir(alpha)。"""
    by_label: Dict[str, List] = {}
    for r in rows:
        by_label.setdefault(by_label_fn(r), []).append(r)
    client_chunks: List[List] = [[] for _ in range(n)]
    for lab, items in by_label.items():
        random.shuffle(items)
        props = [max(1e-6, random.gammavariate(alpha, 1.0)) for _ in range(n)]
        s = sum(props)
        props = [p / s for p in props]
        cuts = [0]
        acc = 0.0
        for p in props[:-1]:
            acc += p
            cuts.append(int(len(items) * acc))
        cuts.append(len(items))
        for c in range(n):
            client_chunks[c].extend(items[cuts[c]:cuts[c + 1]])
    return client_chunks


def holdout_eval(rows: List, ratio: float) -> tuple:
    if ratio <= 0:
        return rows, []
    idx = list(range(len(rows)))
    random.shuffle(idx)
    n_hold = int(len(idx) * ratio)
    hold = [rows[i] for i in idx[:n_hold]]
    keep = [rows[i] for i in idx[n_hold:]]
    return keep, hold


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="训练集 JSON")
    p.add_argument("--eval", default="", help="全局 eval 集（DATA-2: 可不共用，每端用 hold-out）")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-clients", type=int, default=2)
    p.add_argument("--method", choices=["uniform", "fixed", "dirichlet"], default="uniform")
    p.add_argument("--ratios", default="", help="fixed 方法的比例，如 0.7,0.3")
    p.add_argument("--alpha", type=float, default=0.5, help="dirichlet alpha（越小越 non-IID）")
    p.add_argument("--holdout-eval-ratio", type=float, default=0.0,
                   help="从每 client 训练集再 holdout 一份本地 eval（DATA-2，0=不用）")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--prefix", default="client", help="输出文件前缀")
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.train))
    print(f"Loaded {len(rows)} train rows from {args.train}")

    if args.method == "uniform":
        chunks = split_uniform(rows, args.num_clients)
    elif args.method == "fixed":
        ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
        if len(ratios) != args.num_clients:
            raise SystemExit(f"--ratios needs {args.num_clients} values, got {len(ratios)}")
        chunks = split_fixed(rows, ratios)
    else:
        chunks = split_dirichlet(rows, args.num_clients, args.alpha, label_of)

    manifest = {
        "split_method": args.method,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "alpha": args.alpha if args.method == "dirichlet" else None,
        "ratios": args.ratios if args.method == "fixed" else None,
        "holdout_eval_ratio": args.holdout_eval_ratio,
        "counts": {},
        "paths": {"train": str(args.train), "eval": str(args.eval) if args.eval else None},
    }
    for cid, chunk in enumerate(chunks):
        train_chunk = chunk
        local_eval = []
        if args.holdout_eval_ratio > 0:
            train_chunk, local_eval = holdout_eval(chunk, args.holdout_eval_ratio)
        train_path = out_dir / f"{args.prefix}{cid}_train.json"
        train_path.write_text(json.dumps(train_chunk, ensure_ascii=False, indent=2), encoding="utf-8")
        eval_path = None
        if local_eval:
            eval_path = out_dir / f"{args.prefix}{cid}_eval.json"
            eval_path.write_text(json.dumps(local_eval, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["counts"][f"client{cid}"] = {
            "train": len(train_chunk),
            "local_eval": len(local_eval),
            "train_path": str(train_path),
            "eval_path": str(eval_path) if eval_path else None,
        }
        print(f"  client{cid}: train={len(train_chunk)} local_eval={len(local_eval)} -> {train_path}")

    mf_path = out_dir / "federated_split_manifest.json"
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {mf_path}")
    print("\nDATA-2 提示：在 nodes.yaml 里给每端 data_path 指向各自 train，eval-path 指向各自 *_eval.json")
    print("Server 只记录 client_eval_loss，不强行平均不可比的 eval（已在 aggregation_server 区分）")


if __name__ == "__main__":
    main()

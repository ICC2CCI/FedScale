#!/usr/bin/env python3
"""PT-DATA：从 FineWeb-Edu 抽样，切成双 ICC 联邦续预训练语料。

默认：**下载 sample/10BT 的 parquet 到本地再抽样**（避开 datasets streaming
打 huggingface.co API，在国内更稳）。也可用 --source streaming。

用法::

    # 推荐（镜像下 1 个 ~2GB parquet，再抽 200M/20M tokens）
    python scripts/prepare_fineweb_edu_pt.py \\
      --hf-endpoint https://hf-mirror.com \\
      --num-parquets 1 \\
      --train-tokens 200000000 --eval-tokens 20000000

    # 已有 parquet
    python scripts/prepare_fineweb_edu_pt.py \\
      --parquet-dir data/cache/fineweb-edu-10bt \\
      --train-tokens 200000000 --eval-tokens 20000000

输出::

    data/splits/fineweb-edu-pt/
      icc1_train.jsonl / icc2_train.jsonl / eval.jsonl
      README.md / manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SEED = 20260831
DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_REVISION = "main"
# sample/10BT 分片命名：000_00000.parquet ...
PARQUET_REL = "sample/10BT/{idx:03d}_00000.parquet"


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return _approx_tokens(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer(path: Optional[str]):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, use_fast=True, legacy=False, local_files_only=True
    )


def resolve_url(hf_endpoint: str, dataset: str, revision: str, rel: str) -> str:
    base = (hf_endpoint or "https://huggingface.co").rstrip("/")
    return f"{base}/datasets/{dataset}/resolve/{revision}/{rel}"


def download_file(url: str, dest: Path, timeout_s: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  skip existing {dest} ({dest.stat().st_size} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    print(f"    -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "fedscale-pt-prep/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(dest)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")


def ensure_parquets(
    cache_dir: Path,
    hf_endpoint: str,
    dataset: str,
    revision: str,
    num_parquets: int,
) -> List[Path]:
    paths: List[Path] = []
    for i in range(num_parquets):
        rel = PARQUET_REL.format(idx=i)
        dest = cache_dir / Path(rel).name
        url = resolve_url(hf_endpoint, dataset, revision, rel)
        download_file(url, dest)
        paths.append(dest)
    return paths


def iter_parquets(paths: List[Path]) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        print(f"  reading {path}")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            col = batch.column(0)
            for i in range(len(col)):
                text = col[i].as_py()
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if len(text) < 32:
                    continue
                yield {"text": text}


def iter_streaming(dataset: str, subset: str, revision: str, seed: int) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {
        "path": dataset,
        "split": "train",
        "streaming": True,
        "revision": revision,
    }
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs).shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row.get("text") or ""
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 32:
            continue
        yield {"text": text}


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_corpus(
    stream: Iterator[Dict[str, Any]],
    tokenizer,
    train_tokens: int,
    eval_tokens: int,
    seed: int,
    log_every: int = 2000,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    rng = random.Random(seed)
    eval_rows: List[Dict[str, str]] = []
    train_rows: List[Dict[str, str]] = []
    eval_tok = 0
    train_tok = 0
    seen = 0
    t0 = time.time()
    target_total = train_tokens + eval_tokens

    for row in stream:
        text = row["text"]
        n = count_tokens(text, tokenizer)
        seen += 1
        need_eval = eval_tok < eval_tokens
        need_train = train_tok < train_tokens
        if not need_eval and not need_train:
            break
        if need_eval and (not need_train or rng.random() < (eval_tokens / max(1, target_total))):
            eval_rows.append({"text": text})
            eval_tok += n
        elif need_train:
            train_rows.append({"text": text})
            train_tok += n
        if seen % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  seen={seen} train_tok={train_tok}/{train_tokens} "
                f"eval_tok={eval_tok}/{eval_tokens} elapsed={elapsed:.0f}s"
            )

    return train_rows, eval_rows, {
        "docs_seen": seen,
        "train_docs": len(train_rows),
        "eval_docs": len(eval_rows),
        "train_tokens": train_tok,
        "eval_tokens": eval_tok,
        "elapsed_s": round(time.time() - t0, 1),
    }


def split_train_iid(
    train_rows: List[Dict[str, str]], seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    idx = list(range(len(train_rows)))
    rng.shuffle(idx)
    mid = len(idx) // 2
    return [train_rows[i] for i in idx[:mid]], [train_rows[i] for i in idx[mid:]]


def write_readme(out_dir: Path, meta: Dict[str, Any]) -> None:
    text = f"""# FineWeb-Edu 联邦续预训练切分（STAGE-PT）

- **dataset**: `{meta['dataset']}`
- **revision**: `{meta['revision']}`
- **source**: `{meta['source']}`
- **seed**: `{meta['seed']}`
- **token_count**: `{meta['token_count_method']}`
- **train tokens（合计）**: {meta['stats']['train_tokens']:,}
- **eval tokens**: {meta['stats']['eval_tokens']:,}
- **icc1 / icc2 docs**: {meta['icc1_docs']:,} / {meta['icc2_docs']:,}
- **切法**: 文档级随机 50/50（IID proxy）

| 文件 | 用途 |
|---|---|
| `icc1_train.jsonl` | ICC1 / client0 |
| `icc2_train.jsonl` | ICC2 / client1 |
| `eval.jsonl` | 共享 hold-out（PPL/CE） |
| `manifest.json` | 元信息 |

详见 `docs/exec-plans/active/2026-10-08-stage-pt-federated-continued-pretrain.md`。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/splits/fineweb-edu-pt"))
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument(
        "--source",
        choices=["parquet", "streaming"],
        default="parquet",
        help="parquet=下分片再读（推荐）；streaming=datasets（易打到 huggingface.co）",
    )
    p.add_argument("--subset", default="sample-10BT", help="仅 streaming 用")
    p.add_argument("--parquet-dir", type=Path, default=Path("data/cache/fineweb-edu-10bt"))
    p.add_argument("--num-parquets", type=int, default=1, help="下载/使用前 N 个 10BT 分片")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--precise-tokens", action="store_true")
    p.add_argument("--train-tokens", type=int, default=200_000_000)
    p.add_argument("--eval-tokens", type=int, default=20_000_000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    args = p.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    tokenizer = None
    if args.precise_tokens:
        if not args.tokenizer_path:
            raise SystemExit("--precise-tokens 需要 --tokenizer-path")
        print(f"Loading tokenizer: {args.tokenizer_path}")
        tokenizer = load_tokenizer(args.tokenizer_path)
        token_method = "tokenizer"
    else:
        print("Token budget uses approx chars/4")
        token_method = "chars/4"

    if args.source == "parquet":
        print(f"Ensuring {args.num_parquets} parquet(s) under {args.parquet_dir}")
        paths = ensure_parquets(
            args.parquet_dir,
            os.environ["HF_ENDPOINT"],
            args.dataset,
            args.revision,
            args.num_parquets,
        )
        stream = iter_parquets(paths)
        source_meta = {"mode": "parquet", "files": [str(x) for x in paths]}
    else:
        print(f"Streaming {args.dataset} subset={args.subset!r}")
        stream = iter_streaming(args.dataset, args.subset, args.revision, args.seed)
        source_meta = {"mode": "streaming", "subset": args.subset}

    print(f"Sampling train={args.train_tokens} eval={args.eval_tokens}")
    train_rows, eval_rows, stats = sample_corpus(
        stream, tokenizer, args.train_tokens, args.eval_tokens, args.seed
    )
    print(f"Sampled: {json.dumps(stats)}")

    icc1, icc2 = split_train_iid(train_rows, args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "icc1_train.jsonl", icc1)
    write_jsonl(out_dir / "icc2_train.jsonl", icc2)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    meta = {
        "dataset": args.dataset,
        "revision": args.revision,
        "source": source_meta,
        "seed": args.seed,
        "tokenizer_path": args.tokenizer_path or None,
        "token_count_method": token_method,
        "train_tokens_target": args.train_tokens,
        "eval_tokens_target": args.eval_tokens,
        "stats": stats,
        "icc1_docs": len(icc1),
        "icc2_docs": len(icc2),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(out_dir, meta)
    print(f"Wrote {out_dir} icc1={len(icc1)} icc2={len(icc2)} eval={len(eval_rows)}")


if __name__ == "__main__":
    main()

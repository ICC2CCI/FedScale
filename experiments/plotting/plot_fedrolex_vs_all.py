"""Plot FedRolex partial training vs S3R11 vs S3 vs S2."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/fedrolex-vs-all.png")

scenarios = [
    ("S2 full upload", "/data/home/qiaoyanchen/liuchao/fedscale/output/s2-federated-full/round_log.json", "red", "o"),
    ("S3 shard20% (no mem)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3-federated-shard20/round_log.json", "purple", "D"),
    ("S3R11 random20%layer+mem", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-layer-random20/round_log.json", "green", "s"),
    ("FedRolex partial-train", "/data/home/qiaoyanchen/liuchao/fedscale/output/fedrolex-partial-training/round_log.json", "blue", "*"),
]

fig, ax = plt.subplots(figsize=(14, 8))
for name, path, color, marker in scenarios:
    d = json.load(open(path))
    rounds = [r["round"] for r in d]
    evals = [r["eval_loss"] for r in d]
    lw = 2.4 if "FedRolex" in name or "S3R11" in name else 1.8
    ms = 10 if "FedRolex" in name else 6
    ax.plot(rounds, evals, color=color, lw=lw, marker=marker, ms=ms, label=f"{name} (final={evals[-1]:.4f})")

ax.set_xlabel("federated round", fontsize=12)
ax.set_ylabel("eval loss", fontsize=12)
ax.set_title("FedRolex Partial Training vs Full-Train-Partial-Upload\nQwen2.5-0.5B, medical flashcards, 20 rounds, 20% layers/round", fontsize=13)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"\n{'scenario':30s} | final   | min")
print("-"*50)
for name, path, color, marker in scenarios:
    d = json.load(open(path))
    print(f"{name:30s} | {d[-1]['eval_loss']:.4f}  | {min(r['eval_loss'] for r in d):.4f}")

"""Plot S3R12v3 (block-uniform) vs S3R12v2 (key-level) vs S3R11 vs S3 vs S2."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-vs-all.png")

NUM_ROUNDS = 20

scenarios = [
    ("S2 full (100%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s2-federated-full/round_log.json", "red", "P", 1260, 25.2),
    ("S3 shard20% (no mem)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3-federated-shard20/round_log.json", "purple", "D", 252, 5.0),
    ("S3R11 random-layer+mem (55%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-layer-random20/round_log.json", "orange", "^", 694, 13.9),
    ("S3R12v2 key-level (true 20%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v2-block-permutation/round_log.json", "blue", "s", 252, 5.0),
    ("S3R12v3 block-uniform (true 20%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-block-uniform/round_log.json", "green", "*", 252, 5.0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [2, 1]})

for name, path, color, marker, mb, total_gb in scenarios:
    d = json.load(open(path))
    rounds = [r["round"] for r in d]
    evals = [r["eval_loss"] for r in d]
    lw = 3.2 if "S3R12v3" in name else (2.6 if "S3R12v2" in name else 1.8)
    ms = 13 if "S3R12v3" in name else 7
    ax1.plot(rounds, evals, color=color, lw=lw, marker=marker, ms=ms, label=f"{name} (final={evals[-1]:.4f})")

ax1.axhline(y=1.0091, color="gray", ls="--", lw=1, alpha=0.5)
ax1.set_xlabel("federated round", fontsize=12)
ax1.set_ylabel("eval loss", fontsize=12)
ax1.set_title("Eval Loss Curves", fontsize=13)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

results = []
for name, path, color, marker, mb, total_gb in scenarios:
    d = json.load(open(path))
    results.append((name, d[-1]["eval_loss"], mb, total_gb))

colors = [s[2] for s in scenarios]
ax2.plot([r[3] for r in results], [r[1] for r in results], color="black", lw=1.5, alpha=0.3, zorder=1)
for i, (name, final, mb, total_gb) in enumerate(results):
    ax2.scatter(total_gb, final, color=colors[i], s=180, marker=scenarios[i][3], zorder=3, edgecolors="black", lw=0.5)
    ax2.annotate(f"{final:.4f}", (total_gb, final), textcoords="offset points", xytext=(8, 10), fontsize=10, fontweight="bold", color=colors[i])

ax2.set_xlabel("total upload budget (GB, 20 rounds)", fontsize=12)
ax2.set_ylabel("final eval loss (round 20)", fontsize=12)
ax2.set_title("Final Loss vs Total Bandwidth", fontsize=13)
ax2.grid(True, alpha=0.3)

fig.suptitle("S3R12v3 (Block-Level Uniform Upload) vs All — True 20% Bandwidth\nQwen2.5-0.5B, medical flashcards, 20 rounds, 2 clients", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"\n{'scenario':40s} | final   | MB/round | total GB | % of S2")
print("-"*75)
for name, path, color, marker, mb, total_gb in scenarios:
    d = json.load(open(path))
    pct = mb / 1260 * 100
    print(f"{name:40s} | {d[-1]['eval_loss']:.4f}  | {mb:4d} MB | {total_gb:5.1f} GB | {pct:3.0f}%")

print(f"\n=== Upload uniformity comparison (S3R12v2 vs S3R12v3) ===")
v2 = json.load(open("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v2-block-permutation/round_log.json"))
v3 = json.load(open("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-block-uniform/round_log.json"))
v2_pcts = [r["pct_of_total"] for r in v2]
v3_pcts = [r["pct_of_total"] for r in v3]
print(f"S3R12v2: min={min(v2_pcts):.1f}%, max={max(v2_pcts):.1f}%, avg={sum(v2_pcts)/len(v2_pcts):.1f}%, range={max(v2_pcts)-min(v2_pcts):.1f}%")
print(f"S3R12v3: min={min(v3_pcts):.1f}%, max={max(v3_pcts):.1f}%, avg={sum(v3_pcts)/len(v3_pcts):.1f}%, range={max(v3_pcts)-min(v3_pcts):.1f}%")

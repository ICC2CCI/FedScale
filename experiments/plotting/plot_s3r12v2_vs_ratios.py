"""Plot S3R12v2 vs S3R11 ratio ablation (10/20/30/40/50%) vs S2 — bandwidth-fair comparison.

S3R11 ratios: non_layer always_on, so actual bandwidth = 43% + ratio*57%
  10% -> 49% actual
  20% -> 55%
  30% -> 60%
  40% -> 66%
  50% -> 72%
S3R12v2: true 20% (non_layer rotates too)
S2: 100%
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v2-vs-s3r11-ratios.png")

NUM_ROUNDS = 20

scenarios = [
    ("S2 full (100%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s2-federated-full/round_log.json", "red", "P", 1260, 25.2),
    ("S3R11 50% (actual 72%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-ratio-0p5/round_log.json", "purple", "D", 907, 18.1),
    ("S3R11 40% (actual 66%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-ratio-0p4/round_log.json", "violet", "s", 830, 16.6),
    ("S3R11 30% (actual 60%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-ratio-0p3/round_log.json", "blue", "o", 753, 15.1),
    ("S3R11 20% (actual 55%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r11-layer-random20/round_log.json", "orange", "^", 694, 13.9),
    ("S3R12v2 (true 20%)", "/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v2-block-permutation/round_log.json", "green", "*", 252, 5.0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [2, 1]})

for name, path, color, marker, mb, total_gb in scenarios:
    d = json.load(open(path))
    rounds = [r["round"] for r in d]
    evals = [r["eval_loss"] for r in d]
    lw = 3.2 if "S3R12v2" in name else (2.4 if "S3R11 20" in name else 1.6)
    ms = 13 if "S3R12v2" in name else 6
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
    ax2.scatter(total_gb, final, color=colors[i], s=150, marker=scenarios[i][3], zorder=3, edgecolors="black", lw=0.5)
    ax2.annotate(f"{final:.4f}", (total_gb, final), textcoords="offset points", xytext=(8, 10), fontsize=10, fontweight="bold", color=colors[i])

ax2.set_xlabel("total upload budget (GB, 20 rounds)", fontsize=12)
ax2.set_ylabel("final eval loss (round 20)", fontsize=12)
ax2.set_title("Final Loss vs Total Bandwidth", fontsize=13)
ax2.grid(True, alpha=0.3)

fig.suptitle("S3R12v2 (True 20%) vs S3R11 Ratio Ablation (10/20/30/40/50%)\nQwen2.5-0.5B, medical flashcards, 20 rounds, 2 clients", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"\n{'scenario':30s} | final   | MB/round | total GB | % of S2")
print("-"*70)
for name, path, color, marker, mb, total_gb in scenarios:
    d = json.load(open(path))
    pct = mb / 1260 * 100
    print(f"{name:30s} | {d[-1]['eval_loss']:.4f}  | {mb:4d} MB | {total_gb:5.1f} GB | {pct:3.0f}%")

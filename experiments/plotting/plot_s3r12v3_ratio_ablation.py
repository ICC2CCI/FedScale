"""Plot S3R12v3 ratio ablation: 5%, 10%, 20%, 30%, 40%, 50% + S2 baseline."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-ratio-ablation.png")

NUM_ROUNDS = 20

ratios = [
    ("5% (H=20)", "5pct", "red", "o", 63, 1.26),
    ("10% (H=10)", "10pct", "orange", "^", 126, 2.52),
    ("20% (H=5)", "block-uniform", "green", "*", 252, 5.0),
    ("30% (H=3)", "30pct", "blue", "s", 378, 7.6),
    ("40% (H=5,2slots)", "40pct", "purple", "D", 504, 10.1),
    ("50% (H=2)", "50pct", "brown", "P", 630, 12.6),
    ("S2 full (100%)", "full", "gray", "X", 1260, 25.2),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [2, 1]})

results = []
for name, tag, color, marker, mb, total_gb in ratios:
    if tag == "full":
        path = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s2-federated-full/round_log.json")
    elif tag == "block-uniform":
        path = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-block-uniform/round_log.json")
    else:
        path = Path(f"/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-ratio-{tag}/round_log.json")
    d = json.load(open(path))
    rounds = [r["round"] for r in d]
    evals = [r["eval_loss"] for r in d]
    lw = 2.8 if "20%" in name or "S2" in name else 1.8
    ms = 10 if "20%" in name else 6
    ax1.plot(rounds, evals, color=color, lw=lw, marker=marker, ms=ms, label=f"{name} (final={evals[-1]:.4f})")
    results.append((name, evals[-1], mb, total_gb))

ax1.axhline(y=1.0091, color="gray", ls="--", lw=1, alpha=0.5)
ax1.set_xlabel("federated round", fontsize=12)
ax1.set_ylabel("eval loss", fontsize=12)
ax1.set_title("Eval Loss Curves", fontsize=13)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

colors = [r[2] for r in ratios]
ax2.plot([r[3] for r in results], [r[1] for r in results], color="black", lw=1.5, alpha=0.3, zorder=1)
for i, (name, final, mb, total_gb) in enumerate(results):
    ax2.scatter(total_gb, final, color=colors[i], s=150, marker=ratios[i][3], zorder=3, edgecolors="black", lw=0.5)
    ax2.annotate(f"{final:.4f}", (total_gb, final), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold", color=colors[i])

ax2.set_xlabel("total upload budget (GB, 20 rounds)", fontsize=12)
ax2.set_ylabel("final eval loss (round 20)", fontsize=12)
ax2.set_title("Final Loss vs Total Bandwidth", fontsize=13)
ax2.grid(True, alpha=0.3)

fig.suptitle("S3R12v3 (Block-Level Uniform Upload) Ratio Ablation\nQwen2.5-0.5B, medical flashcards, 20 rounds, 2 clients", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"\n{'ratio':25s} | final   | MB/round | total GB | gap vs S2")
print("-"*60)
for name, final, mb, total_gb in results:
    gap = final - 1.0091
    print(f"{name:25s} | {final:.4f}  | {mb:4d} MB | {total_gb:5.1f} GB | +{gap:.4f}")

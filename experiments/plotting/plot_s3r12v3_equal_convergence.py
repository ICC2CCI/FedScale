"""Plot equal-convergence bandwidth comparison: total upload to reach same eval loss."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/home/qiaoyanchen/liuchao/fedscale/output/s3r12v3-equal-convergence.png")

ratios = [
    ("5%", "s3r12v3-ratio-5pct", 63, "red", "o"),
    ("10%", "s3r12v3-ratio-10pct", 126, "orange", "^"),
    ("20%", "s3r12v3-block-uniform", 252, "green", "*"),
    ("30%", "s3r12v3-ratio-30pct", 378, "blue", "s"),
    ("40%", "s3r12v3-ratio-40pct", 504, "purple", "D"),
    ("50%", "s3r12v3-ratio-50pct", 630, "brown", "P"),
    ("100% (S2)", "s2-federated-full", 1260, "gray", "X"),
]

base_dir = Path("/data/home/qiaoyanchen/liuchao/fedscale/output")
targets = [1.15, 1.12, 1.10, 1.08, 1.05]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left: stacked bar chart — total upload to reach each target
import numpy as np
x = np.arange(len(targets))
width = 0.11
colors = [r[3] for r in ratios]
for i, (name, dirname, mb, color, marker) in enumerate(ratios):
    d = json.load(open(base_dir / dirname / "round_log.json"))
    vals = []
    for t in targets:
        reached = None
        for r in d:
            if r["eval_loss"] <= t:
                reached = r["round"]
                break
        vals.append(reached * mb / 1000 if reached else 0)
    bars = ax1.bar(x + i * width - width * 3, vals, width, label=name, color=color, alpha=0.85,
                   edgecolor="black", lw=0.3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7, rotation=90)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, 0.3, "✗",
                     ha="center", va="bottom", fontsize=8, color="gray")

ax1.set_xticks(x)
ax1.set_xticklabels([f"eval ≤ {t:.2f}" for t in targets], fontsize=11)
ax1.set_ylabel("total upload (GB) to reach target", fontsize=12)
ax1.set_title("Total Upload to Reach Equal Convergence", fontsize=13)
ax1.legend(loc="upper left", fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 28)

# Right: line chart — bandwidth vs target, with 1.10 highlighted
for name, dirname, mb, color, marker in ratios:
    d = json.load(open(base_dir / dirname / "round_log.json"))
    xs, ys = [], []
    for t in targets:
        reached = None
        for r in d:
            if r["eval_loss"] <= t:
                reached = r["round"]
                break
        if reached:
            xs.append(t)
            ys.append(reached * mb / 1000)
    lw = 3.0 if name in ("10%", "20%", "30%") else 1.8
    ms = 11 if name in ("10%", "20%", "30%") else 7
    ax2.plot(xs, ys, color=color, lw=lw, marker=marker, ms=ms, label=name)

ax2.axvline(x=1.10, color="red", ls="--", lw=1.5, alpha=0.5)
ax2.text(1.103, 1, "recommended\nthreshold 1.10", fontsize=9, color="red", va="bottom")
ax2.set_xlabel("target eval loss", fontsize=12)
ax2.set_ylabel("total upload (GB)", fontsize=12)
ax2.set_title("Bandwidth vs Convergence Target", fontsize=13)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

fig.suptitle("S3R12v3 Equal-Convergence Bandwidth Comparison\nTotal upload to reach same eval loss — lower is better",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")

print(f"\n=== Equal-convergence bandwidth (GB) ===")
print(f"{'target':8s} |", " | ".join(f"{r[0]:>10s}" for r in ratios))
print("-"*85)
for t in targets:
    row = f"{t:.2f}     |"
    for name, dirname, mb, color, marker in ratios:
        d = json.load(open(base_dir / dirname / "round_log.json"))
        reached = None
        for r in d:
            if r["eval_loss"] <= t:
                reached = r["round"]
                break
        if reached:
            gb = reached * mb / 1000
            row += f" {gb:8.1f}GB |"
        else:
            row += f"      ✗    |"
    print(row)

print(f"\n=== Best ratio at each target ===")
for t in targets:
    best_name, best_gb = None, 1e9
    for name, dirname, mb, color, marker in ratios:
        d = json.load(open(base_dir / dirname / "round_log.json"))
        reached = None
        for r in d:
            if r["eval_loss"] <= t:
                reached = r["round"]
                break
        if reached:
            gb = reached * mb / 1000
            if gb < best_gb:
                best_gb = gb
                best_name = name
    s2_reached = None
    for r in json.load(open(base_dir / "s2-federated-full" / "round_log.json")):
        if r["eval_loss"] <= t:
            s2_reached = r["round"]
            break
    s2_gb = s2_reached * 1260 / 1000 if s2_reached else 0
    savings = (1 - best_gb / s2_gb) * 100 if s2_gb else 0
    print(f"  {t:.2f}: {best_name:>6s} ({best_gb:.1f} GB) vs S2 ({s2_gb:.1f} GB) → save {savings:.0f}%")

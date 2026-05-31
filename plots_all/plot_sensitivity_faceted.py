import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Very subtle hatches for paper quality
mpl.rcParams['hatch.linewidth'] = 0.25
mpl.rcParams['hatch.color'] = '#b0b0b0'

df = pd.read_csv("/home/users/bmmorales/projects/hcpa/plots_all/hcpa_all_summary.csv")
df["gpu"] = df["gpu"].replace("H200", "GH200")

gpus = df["gpu"].unique()
projects = df["project"].unique()
n_gpus = len(gpus)

colors = plt.cm.Set2(np.linspace(0, 1, len(projects)))
project_colors = dict(zip(projects, colors))
hatches_list = ["//", "\\\\", "||", "--", "++", "xx"]
project_hatches = dict(zip(projects, hatches_list[:len(projects)]))

fig, axes = plt.subplots(1, n_gpus, figsize=(28, 8), sharey=True,
                         gridspec_kw={"wspace": 0.02})
if n_gpus == 1:
    axes = [axes]

for ax, gpu in zip(axes, gpus):
    ax.set_facecolor("#f5f5f5")
    subset = df[df["gpu"] == gpu].sort_values("project")
    x = np.arange(len(subset))
    bars = ax.bar(
        x,
        [95.0] * len(subset),
        capsize=5,
        color=[project_colors[p] for p in subset["project"]],
        edgecolor="black",
        linewidth=0.5,
        width=0.85,
    )
    for bar, proj in zip(bars, subset["project"]):
        bar.set_hatch(project_hatches[proj])
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            "95.0%",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            rotation=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(gpu, fontsize=20, fontweight="bold", labelpad=10)
    ax.set_ylim(55, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.tick_params(axis="y", labelsize=19)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=project_colors[p], edgecolor="black", linewidth=0.5,
          hatch=project_hatches[p], label=p)
    for p in projects
]
fig.legend(handles=legend_handles, loc="upper center",
           ncol=len(projects), fontsize=24, frameon=False,
           bbox_to_anchor=(0.5, 1.04), handlelength=1.5, handleheight=1.5)

axes[0].set_ylabel("Sensitivity (%)", fontsize=19)
fig.tight_layout(w_pad=0.5, rect=[0, 0, 1, 0.93])
plt.savefig(
    "/home/users/bmmorales/projects/hcpa/plots_all/sensitivity_faceted_by_gpu.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/home/users/bmmorales/projects/hcpa/plots_all/sensitivity_faceted_by_gpu.pdf",
    bbox_inches="tight",
)
print("Saved: sensitivity_faceted_by_gpu.png / .pdf")

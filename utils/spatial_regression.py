# coef_boxplots.py
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
INPUT_CSV = "../out/spatial_regression_results.csv"   # adjust path
OUTFIG = "../figures/coef_boxplot_grouped.png"

# predictors to plot (these must match the names used in coef_json)
predictors = [
    "Ethno.cultural.Composition",
    "Economic.Dependency",
    "CISV.Scores",
    # include whichever area variable(s) you want,
    # we'll map them to a common label per outcome below
]
# label map for nicer Y ticks
label_map = {
    "Ethno.cultural.Composition": "Ethno-cultural Composition",
    "Economic.Dependency": "Economic Dependency",
    "CISV.Scores": "CISV Score",
    "total_area_km2_da": "Total area (DA)",
    "total_area_km2_road": "Total area (Road)"
}
# colors for DA vs Road
COL_DA = "#1f77b4"   # blue
COL_ROAD = "#ff7f0e" # orange
# ---------------------------------

df = pd.read_csv(INPUT_CSV, dtype={"CSDUID": str}, keep_default_na=False)
# ensure missing coef_json are treated as NA
df["coef_json"] = df["coef_json"].replace("", pd.NA)

# parse coef_json safely
def parse_coef_json(s):
    if pd.isna(s):
        return {}
    # some rows may have quotes escaped (they are JSON strings already)
    try:
        # if the field is like: "{"rho":0.14, ...}" -> json.loads ok
        parsed = json.loads(s)
        # ensure keys are strings and values numeric where possible
        return {str(k): (float(v) if (v is not None and not isinstance(v, dict)) else v) for k, v in parsed.items()}
    except Exception:
        # try replacing doubled quotes etc. (be lenient)
        try:
            return json.loads(s.replace("''", '"').replace("'", '"'))
        except Exception:
            return {}

df["parsed_coefs"] = df["coef_json"].map(parse_coef_json)

# Expand into long table: one row per (CSDUID, outcome, predictor, coef)
long_rows = []
for _, row in df.iterrows():
    outcome = row["outcome"]                  # e.g. 'canopy_proportion_da' or '_road'
    n_DA = row.get("n_DA", None)
    parsed = row["parsed_coefs"] or {}
    for k, v in parsed.items():
        # skip spatial params like "rho" or "lambda" unless user wants them
        if k in ["rho", "lambda"]:
            continue
        # keep intercept optionally; here we skip intercept for plotting predictors
        if k == "(Intercept)":
            continue
        long_rows.append({
            "CSDUID": row["CSDUID"],
            "outcome": outcome,
            "predictor": k,
            "coef": v,
            "n_DA": n_DA
        })

long_df = pd.DataFrame(long_rows)

# Map the two area predictors to separate outcomes if needed:
# We want predictors list to be consistent across outcomes. For area, the coef keys differ:
# - for canopy_proportion_da outcome the area var is 'total_area_km2_da'
# - for canopy_proportion_road outcome the area var is 'total_area_km2_road'
# We'll build a canonical list of predictors to display:
display_predictors = list(predictors) + ["total_area_km2_da", "total_area_km2_road"]

# Build plotting matrix: for each predictor in a canonical order,
# collect coefficients for DA and for Road (based on outcome value)
# Determine which outcome names correspond to DA vs Road:
# we assume outcomes contain '_da' and '_road' suffixes (as in your data)
is_da_mask = long_df["outcome"].str.contains("_da", case=False, na=False)
is_road_mask = long_df["outcome"].str.contains("_road", case=False, na=False)

# For plotting we need predictor ordering (top to bottom)
plot_predictors = [p for p in display_predictors if p in long_df["predictor"].unique()]

# Create figure
fig_w = 7.0
fig_h = max(4.0, 0.6 * len(plot_predictors))  # scale height with number of predictors
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Y positions: one position per predictor, but we will offset two boxes per predictor
n = len(plot_predictors)
y_pos = np.arange(n)  # 0..n-1 (we'll flip later so first item is on top)
offset = 0.2

# We'll collect patch handles for legend
legend_handles = []

# Prepare lists to plot boxplots manually (matplotlib.boxplot uses vertical by default; we'll use horizontal)
for i, pred in enumerate(plot_predictors):
    # get coef arrays
    da_coefs = long_df.loc[ (long_df["predictor"] == pred) & (is_da_mask), "coef"].dropna().astype(float).values
    road_coefs = long_df.loc[ (long_df["predictor"] == pred) & (is_road_mask), "coef"].dropna().astype(float).values

    # central y coordinate (we draw DA slightly left/up, Road slightly right/down)
    center = y_pos[i]

    # positions for the two boxplots (horizontal)
    pos_da = center + (-offset)
    pos_road = center + (offset)

    # draw boxplots horizontally. We will call boxplot twice with positions specified.
    # Note: matplotlib's boxplot for horizontal boxes: use vert=False and positions argument
    if len(da_coefs) > 0:
        bp_da = ax.boxplot(da_coefs, positions=[pos_da], vert=False, widths=0.35,
                           patch_artist=True, manage_ticks=False)
        # style DA box
        for elem in ["boxes", "medians", "whiskers", "caps"]:
            for patch in bp_da[elem]:
                patch.set_color(COL_DA)
        for patch in bp_da["boxes"]:
            patch.set_facecolor(COL_DA)
        # add scatter of individual points (jittered)
        jitter = (np.random.rand(len(da_coefs)) - 0.5) * 0.08
        ax.scatter(da_coefs, np.full_like(da_coefs, pos_da) + jitter, s=12, color=COL_DA, alpha=0.6, zorder=5)
    if len(road_coefs) > 0:
        bp_road = ax.boxplot(road_coefs, positions=[pos_road], vert=False, widths=0.35,
                             patch_artist=True, manage_ticks=False)
        for elem in ["boxes", "medians", "whiskers", "caps"]:
            for patch in bp_road[elem]:
                patch.set_color(COL_ROAD)
        for patch in bp_road["boxes"]:
            patch.set_facecolor(COL_ROAD)
        jitter = (np.random.rand(len(road_coefs)) - 0.5) * 0.08
        ax.scatter(road_coefs, np.full_like(road_coefs, pos_road) + jitter, s=12, color=COL_ROAD, alpha=0.6, zorder=5)

# Draw vertical line at 0 (no effect)
ax.axvline(0, color="0.3", linewidth=0.9, linestyle="--", zorder=2)

# Y ticks in middle of DA/Road pair
ax.set_yticks(y_pos)
ytick_labels = [label_map.get(v, v) for v in plot_predictors]
ax.set_yticklabels(ytick_labels, fontsize=9)

# Labels, title
ax.set_xlabel("Standardised Coefficient (β)", fontsize=10)
ax.set_title("Distribution of Model Coefficients by Predictor and Outcome", fontsize=11)

# legend: create custom legend elements
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COL_DA, edgecolor=COL_DA, label="DA Canopy"),
                   Patch(facecolor=COL_ROAD, edgecolor=COL_ROAD, label="Street Tree Canopy")]
ax.legend(handles=legend_elements, frameon=False, fontsize=9, loc="upper right")

# Style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#EAEAEA")
ax.set_axisbelow(True)

# invert y so first predictor appears at top (like many publication plots)
ax.invert_yaxis()
fig.tight_layout()

# ensure output dir exists
outpath = Path(OUTFIG)
outpath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
print(f"Wrote: {outpath}")

plt.show()
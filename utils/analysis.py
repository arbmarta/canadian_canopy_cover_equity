import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from libpysal.weights import Queen
from spreg import OLS, ML_Lag, ML_Error, GM_Combo
from esda.moran import Moran
import warnings
from libpysal import weights

da_df = pd.read_csv("../data/dissemination_areas/dissemination_areas.csv")
rn_df = pd.read_csv("../data/road_network/road_network.csv")

rn_df = rn_df.drop(columns=["LANDAREA", "CSDUID", "CSDNAME"])
df = da_df.merge(rn_df, on="DAUID", how="left", suffixes=("_da", "_road"))

pop_dens = pd.read_csv('../data/population_density/98100015.csv')
cimd = pd.read_csv("../data/can_scores_quintiles_csv-eng/can_scores_quintiles_EN.csv")
cisr = pd.read_csv('../data/cisr-eng/cisr_scores_quintiles-eng.csv')
cisv = pd.read_csv('../data/cisv-eng/cisv_scores_quintiles-eng.csv')
census_indep_vars = pd.read_csv("../data/Canadian_urban_forest_census_independent_variables.csv")

df = (
    df
    .merge(pop_dens, on="DAUID", how="left", suffixes=("", "_da"))
    .merge(cimd, on="DAUID", how="left")
    .merge(cisr, on="DAUID", how="left")
    .merge(cisv, on="DAUID", how="left")
    .merge(census_indep_vars, on="CSDUID", how="left")
)

print(df.columns)

# Encode binary variables
df['in_eab_area_2024'] = df['in_eab_area_2024'].map({'Yes': 1, 'No': 0})
df['in_eab_area_2025'] = df['in_eab_area_2025'].map({'Yes': 1, 'No': 0})

df.to_csv("../data/combined_dataset.csv", index=False)


## ----------------------------------------------- CALCULATE GINI INDEX ------------------------------------------------
#region

def gini(values):
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or values.sum() == 0:
        return np.nan
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

# National Gini (all DAUIDs, no threshold)
national_gini_da   = gini(df["canopy_area_km2_da"].values)
national_gini_road = gini(df["canopy_area_km2_road"].values)

# Per-CSD Gini (≥ 30 DAUIDs only)
counts = df.groupby("CSDUID").size()
valid_csds = counts[counts >= 30].index
df_filtered = df[df["CSDUID"].isin(valid_csds)]

gini_da   = df_filtered.groupby("CSDUID")["canopy_area_km2_da"].apply(gini)
gini_road = df_filtered.groupby("CSDUID")["canopy_area_km2_road"].apply(gini)

gini_df = pd.DataFrame({"DA": gini_da, "Road": gini_road}).dropna()
gini_df["Difference"] = gini_df["Road"] - gini_df["DA"]

#  Styling 
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
    "axes.spines.right": False,
    "axes.spines.top": False,
})

COL_DA   = "#2D6A4F"
COL_ROAD = "#D4A017"
COL_POS  = "#E76F51"
COL_NEG  = "#457B9D"

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 4.2),
    gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.25},
)

# ---------------- Panel A: Distribution ----------------
vals_da   = gini_df["DA"].values
vals_road = gini_df["Road"].values
positions = [1, 2]

vp1 = ax1.violinplot(vals_da, positions=[1], widths=0.9,
                     showmeans=False, showmedians=False, showextrema=False)
vp2 = ax1.violinplot(vals_road, positions=[2], widths=0.9,
                     showmeans=False, showmedians=False, showextrema=False)

for pc, col in zip([vp1["bodies"][0], vp2["bodies"][0]], [COL_DA, COL_ROAD]):
    pc.set_facecolor(col)
    pc.set_edgecolor(None)
    pc.set_alpha(0.35)

ax1.boxplot([vals_da, vals_road], positions=positions, widths=0.25,
            patch_artist=True,
            boxprops=dict(facecolor="none", linewidth=1),
            medianprops=dict(color="black", linewidth=1.4),
            showfliers=False)

ax1.hlines(national_gini_da,   0.6, 1.4, colors=COL_DA,   linestyles="--", linewidth=1.4)
ax1.hlines(national_gini_road, 1.6, 2.4, colors=COL_ROAD, linestyles="--", linewidth=1.4)

ax1.set_ylim(0, 1)
ax1.set_xticks(positions)
ax1.set_xticklabels(["Dissemination Area\nCanopy Cover", "Street Tree\nCanopy Cover"])
ax1.set_ylabel("Gini Index")
ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#EAEAEA")
ax1.set_axisbelow(True)

# ---------------- Panel B: Differences ----------------
diff = gini_df["Difference"].sort_values()
x = np.arange(len(diff))
bar_colors = [COL_POS if v >= 0 else COL_NEG for v in diff]

ax2.bar(x, diff.values, color=bar_colors, width=1.2, linewidth=0)
ax2.axhline(0, color="0.2", linewidth=0.8, linestyle="-", alpha=0.7)

ax2.set_xlim(-0.5, len(diff) - 0.5)
ax2.set_ylim(-0.55, 0.08)
ax2.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.05])
ax2.set_ylabel("Difference in Gini Index")
ax2.set_xticks([])
ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#EAEAEA")
ax2.set_axisbelow(True)

ymin, ymax = ax2.get_ylim()
x_center = (len(diff) - 1) / 2.0
textbox = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.6)

if ymax > 0.0:
    text_y = (0 + ymax) / 2.0
    ax2.annotate(
        "Overall Tree Canopy\nMore Equitable",
        xy=(len(diff) - 5, text_y),
        xytext=(98, text_y),
        ha="center", va="center",
        fontsize=9, fontweight="semibold",
        color='black', zorder=10,
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=0.8,
            connectionstyle="arc3,rad=0"
        )
    )

idx = int(np.floor(129 / 2))
y_value = diff.values[idx]
print(f"x = {idx}, y = {y_value}")

if ymin < 0.0:
    ax2.text((129/2), (y_value / 2), "Street Tree Canopy\nMore Equitable",
             ha="center", va="center", fontsize=12, fontweight="semibold",
             color='white', zorder=10)

fig.tight_layout()
fig.savefig("../figures/Figure 2.png", dpi=600, bbox_inches="tight")
plt.show()

print(f"> 0: {(diff > 0).sum()} | < 0: {(diff < 0).sum()} | = 0: {(diff == 0).sum()}")
print(f"National Gini — DA: {national_gini_da:.4f} | Road: {national_gini_road:.4f} | Diff: {national_gini_road - national_gini_da:+.4f}")
print(f"CSDs analysed: {len(gini_df)}")

#endregion

## --------------------------------------------- CALCULATE NATIONAL EQUITY ---------------------------------------------
#region

# Define variables
canopy_cover = ['canopy_proportion_da', 'canopy_proportion_road']
area = ['total_area_km2_da', 'total_area_km2_road']

indep_vars = [
    #'Population Density (sq km)_x', 'Population, 2021_x',
    'coverage_pct', 'in_eab_area_2024',
    'avg_annual_precip_mm', 'avg_annual_frost_free_days',
    'Ethno-cultural Composition', 'Economic Dependency',
    'CISV Scores'
]

# Variables to standardise (continuous only — binary left as-is)
continuous_vars = [
    #'Population Density (sq km)_x', 'Population, 2021_x',
    'coverage_pct',
    'avg_annual_precip_mm', 'avg_annual_frost_free_days',
    'Ethno-cultural Composition', 'Economic Dependency',
    'CISV Scores'
]
binary_vars = ['in_eab_area_2024']

coef_results = {}

for y_var, area_var in zip(canopy_cover, area):

    print("\n" + "="*60)
    print(f"Regression for: {y_var}")
    print("="*60)

    # Collect all needed columns and drop missing
    cols = [y_var, area_var] + indep_vars
    data = df[cols].dropna().copy()

    #  Standardise continuous predictors 
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[continuous_vars] = scaler.fit_transform(data[continuous_vars])
    data_scaled[area_var] = scaler.fit_transform(data[[area_var]])  # scale area separately

    # Build X and y
    X = data_scaled[indep_vars + [area_var]]
    y = data_scaled[y_var]  # also scale y so coefficients are fully standardised (betas)

    X = sm.add_constant(X)

    #  OLS with HC3 robust standard errors 
    model = sm.OLS(y, X).fit(cov_type='HC3')
    print(model.summary())

    #  Assumption Checks 
    residuals = model.resid
    fitted = model.fittedvalues

    print("\n--- Assumption Checks ---")

    # 1. Linearity
    plt.scatter(fitted, residuals, alpha=0.3, s=5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Residuals vs Fitted ({y_var})")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()

    # 2. Normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test p-value: {shapiro_p:.4f}")

    sm.qqplot(residuals, line='45')
    plt.title(f"QQ Plot ({y_var})")
    plt.tight_layout()
    plt.show()

    # 3. Homoscedasticity
    bp_test = sms.het_breuschpagan(residuals, model.model.exog)
    print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")

    # 4. Multicollinearity (VIF computed on unscaled X for comparability)
    X_unscaled = sm.add_constant(data[indep_vars + [area_var]])
    vif_data = pd.DataFrame({
        "Variable": X_unscaled.columns,
        "VIF": [variance_inflation_factor(X_unscaled.values, i)
                for i in range(X_unscaled.shape[1])]
    })
    print("\nVIF (computed on unscaled predictors):")
    print(vif_data.to_string(index=False))

    # 5. Independence
    print(f"\nDurbin-Watson: {sm.stats.durbin_watson(residuals):.4f}")

    # Store results for combined coefficient plot (significant only)
    coef_results[y_var] = pd.DataFrame({
        "coef":  model.params,
        "lower": model.conf_int()[0],
        "upper": model.conf_int()[1],
        "pval":  model.pvalues,
    }).drop(index="const")

#  Combined Coefficient Plot (significant predictors only) ─
COL_DA   = "#2D6A4F"
COL_ROAD = "#D4A017"

label_map = {
    'Population, 2021_x':           "Population",
    'Population Density (sq km)_x': "Population Density (sq km)",
    "coverage_pct":                 "Municipal Canopy Coverage (%)",
    "in_eab_area_2024":             "EAB Infestation Area",
    "avg_annual_precip_mm":         "Annual Precipitation (mm)",
    "avg_annual_frost_free_days":   "Frost-Free Days",
    "Ethno-cultural Composition":   "Ethno-cultural Composition",
    "Economic Dependency":          "Economic Dependency",
    "CISV Scores":                  "Social Vulnerability (CISV)",
    "total_area_km2_da":            "DA Area (km²)",
    "total_area_km2_road":          "Road Area (km²)",
}

da_sig   = coef_results["canopy_proportion_da"][coef_results["canopy_proportion_da"]["pval"] < 0.05]
road_sig = coef_results["canopy_proportion_road"][coef_results["canopy_proportion_road"]["pval"] < 0.05]

# Union of significant variables, preserving indep_vars order
all_vars = indep_vars + area
sig_vars = [v for v in all_vars if v in da_sig.index or v in road_sig.index]

y_pos  = np.arange(len(sig_vars))
offset = 0.18  # vertical offset between the two model dots

fig_c, ax_c = plt.subplots(figsize=(8, 0.6 * len(sig_vars) + 1.5))

for i, var in enumerate(sig_vars):
    for j, (results, color, label) in enumerate([
        (da_sig,   COL_DA,   "DA Canopy"),
        (road_sig, COL_ROAD, "Street Tree Canopy"),
    ]):
        yo = y_pos[i] + (j - 0.5) * offset
        if var in results.index:
            row = results.loc[var]
            ax_c.errorbar(
                row["coef"], yo,
                xerr=[[row["coef"] - row["lower"]], [row["upper"] - row["coef"]]],
                fmt="o", color=color, markersize=6, linewidth=1.4, capsize=3,
                label=label if i == 0 else "_nolegend_",
                zorder=4
            )

ax_c.axvline(0, color="0.3", linewidth=0.9, linestyle="--", zorder=2)

ytick_labels = [label_map.get(v, v) for v in sig_vars]
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(ytick_labels, fontsize=9)
ax_c.set_xlabel("Standardised Coefficient (β)", fontsize=10)
ax_c.set_title("Significant Predictors of Canopy Cover (p < 0.05)", fontsize=10)
ax_c.legend(frameon=False, fontsize=9)
ax_c.spines[["top", "right"]].set_visible(False)
ax_c.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#EAEAEA")
ax_c.set_axisbelow(True)

fig_c.tight_layout()
fig_c.savefig("../figures/coef_plot_combined.png", dpi=300, bbox_inches="tight")
plt.show()

#endregion

## ----------------------------------- IDENTIFY SPATIAL LAG AND SPATIAL ERROR BY CSD -----------------------------------
#region

# Load shapefile and ensure string keys
gdf = gpd.read_file("../data/dissemination_areas/dissemination_areas_2021.shp")
gdf["DAUID"] = gdf["DAUID"].astype(str)
df["DAUID"]  = df["DAUID"].astype(str)

# Merge attribute data onto geodataframe (avoid duplicating identical columns)
cols_to_merge = [c for c in df.columns if c not in gdf.columns or c == "DAUID"]
gdf = gdf.merge(df[cols_to_merge], on="DAUID", how="left")

# Spatial model variables (small set you've chosen)
spatial_indep_vars = [
    'Ethno-cultural Composition', 'Economic Dependency',
    'CISV Scores'
]
# For scaling we will derive per-CSD the actual continuous cols present
spatial_canopy_vars = ['canopy_proportion_da', 'canopy_proportion_road']
spatial_area_vars   = ['total_area_km2_da',    'total_area_km2_road']

# Identify valid CSDs (>= 30 DAs)
csd_counts = gdf.groupby("CSDUID").size()
valid_csds = csd_counts[csd_counts >= 30].index.tolist()
print(f"\nCSDs with >= 30 DAs: {len(valid_csds)}")

# Spatial diagnostics table by CSD
results = []

for csd_id in valid_csds:
    csd_gdf = gdf[gdf["CSDUID"] == csd_id].copy().reset_index(drop=True)
    csd_name = csd_gdf.get("CSDNAME", pd.Series([str(csd_id)])).iloc[0]

    for y_var, area_var in zip(spatial_canopy_vars, spatial_area_vars):

        needed = [y_var, area_var] + spatial_indep_vars
        sub = csd_gdf[needed + ["geometry"]].dropna().copy().reset_index(drop=True)

        if len(sub) < 30:
            continue

        present_predictors = [c for c in spatial_indep_vars if c in sub.columns]
        if not present_predictors or area_var not in sub.columns:
            continue

        sub_s = sub.copy()
        scale_cols = [c for c in present_predictors if np.issubdtype(sub[c].dtype, np.number)]
        if np.issubdtype(sub[area_var].dtype, np.number):
            scale_cols += [area_var]
        if scale_cols:
            sub_s[scale_cols] = StandardScaler().fit_transform(sub[scale_cols])
        if np.issubdtype(sub_s[y_var].dtype, np.number):
            sub_s[y_var] = StandardScaler().fit_transform(sub[[y_var]])
        else:
            continue

        y = sub_s[y_var].astype(float).values.reshape(-1, 1)
        X_cols = present_predictors + [area_var]
        X_df = sub_s[X_cols].copy()
        X_df['const'] = 1.0
        try:
            X_df = X_df.astype(float)
        except Exception:
            continue

        # Drop zero-variance columns
        tiny_eps = 1e-8
        zero_var_cols = [c for c in X_df.columns
                         if X_df[c].nunique(dropna=True) <= 1 or np.nanstd(X_df[c].values) < tiny_eps]
        X_df = X_df.drop(columns=[c for c in zero_var_cols if c != 'const'])
        X_order = ['const'] + [c for c in X_cols if c in X_df.columns]
        if X_order == ['const']:
            continue

        X_array = X_df[X_order].values

        try:
            w = Queen.from_dataframe(sub, use_index=False, silence_warnings=True)
            w.transform = "r"
        except Exception:
            continue
        if w.n != len(sub):
            continue

        try:
            ols = OLS(y, X_array, w=w, spat_diag=True,
                      name_y=y_var, name_x=X_order, name_ds=csd_name)
        except Exception as e:
            print(f"  [{csd_name} | {y_var}] OLS error: {e}")
            continue

        results.append({
            "CSD":           csd_name,
            "CSDUID":        csd_id,
            "outcome":       y_var,
            "n_DA":          len(sub),
            "LM Lag":        round(ols.lm_lag[0], 4)   if hasattr(ols, "lm_lag")   else np.nan,
            "LM Lag p":      round(ols.lm_lag[1], 4)   if hasattr(ols, "lm_lag")   else np.nan,
            "LM Error":      round(ols.lm_error[0], 4) if hasattr(ols, "lm_error") else np.nan,
            "LM Error p":    round(ols.lm_error[1], 4) if hasattr(ols, "lm_error") else np.nan,
            "RLM Lag":       round(ols.rlm_lag[0], 4)  if hasattr(ols, "rlm_lag")  else np.nan,
            "RLM Lag p":     round(ols.rlm_lag[1], 4)  if hasattr(ols, "rlm_lag")  else np.nan,
            "RLM Error":     round(ols.rlm_error[0], 4)if hasattr(ols, "rlm_error")else np.nan,
            "RLM Error p":   round(ols.rlm_error[1], 4)if hasattr(ols, "rlm_error")else np.nan,
            "LM SAC": round(ols.lm_sarma[0], 4) if hasattr(ols, "lm_sarma") else np.nan,
            "LM SAC p": round(ols.lm_sarma[1], 4) if hasattr(ols, "lm_sarma") else np.nan,
        })


diag_df = pd.DataFrame(results)

def classify_model(row):
    lag_sig   = row["LM Lag p"]   < 0.05
    error_sig = row["LM Error p"] < 0.05

    if not lag_sig and not error_sig:
        return "OLS"
    elif error_sig and not lag_sig:
        return "SEM"
    elif lag_sig and not error_sig:
        return "SLM"
    else:  # both significant — go to robust tests
        rlag_sig   = row["RLM Lag p"]   < 0.05
        rerror_sig = row["RLM Error p"] < 0.05

        if rerror_sig and not rlag_sig:
            return "SEM"
        elif rlag_sig and not rerror_sig:
            return "SLM"
        elif not rlag_sig and not rerror_sig:
            return "SAC"
        else:  # both robust significant — pick largest of LM Lag, LM Error, LM SAC
            stats = {}
            if row["LM Lag p"] < 0.05: stats["SLM"] = row["LM Lag"]
            if row["LM Error p"] < 0.05: stats["SEM"] = row["LM Error"]
            if row["LM SAC p"] < 0.05: stats["SAC"] = row["LM SAC"]

            if stats:
                winner = max(stats, key=stats.get)
                return f"{winner} (all significant)"
            else:
                return "error"  # fallback if somehow none are significant

diag_df["test_type"] = diag_df.apply(classify_model, axis=1)

col_order = ["CSDUID", "outcome", "n_DA",
             "LM Lag", "LM Lag p", "LM Error", "LM Error p",
             "RLM Lag", "RLM Lag p", "RLM Error", "RLM Error p",
             "LM SAC", "LM SAC p", "test_type"]

print(diag_df[col_order].to_string(index=False))
diag_df.to_csv("../data/spatial_diagnostics.csv", index=False)



#endregion
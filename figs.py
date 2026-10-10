import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = json.load(open("pool_out/final_results.json")); A = json.load(open("pool_out/age40_results.json"))
COL = {"Age only": "#9AA5B1", "Age + sex": "#6B7A8F", "Original 9 variables (refit)": "#7FB3D5", "Compact 12-variable Cox": "#1F618D",
       "Elastic-net Cox (61 candidates)": "#17202A", "Gradient boosting": "#CA6F1E"}
LAB = {"Original 9 variables (refit)": "Original 9-variable model"}
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)
# ---- Fig 1 cohort flow
fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=300); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def box(x, y, w, h, t, fc="#EAF2F8", ec="#1F618D", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
box(2.3, 8.7, 5.4, 1.1, "NHANES 2001-2008 (four 2-year cycles)\nlinked to NDI mortality through 2019\nn = 23,377 mortality-eligible participants", bold=False)
arr(5, 8.7, 5, 7.9)
box(2.3, 6.8, 5.4, 1.1, "Adults aged 40 years or older\nn = 13,847 (3,365 deaths within 10 years)\n(no one excluded for < 120 months of follow-up)")
arr(3.2, 6.8, 2.4, 5.8); arr(6.8, 6.8, 7.6, 5.8)
box(0.2, 4.4, 4.4, 1.4, "Development cohort\nNHANES 2001-2002, 2003-2004, 2005-2006\nn = 9,828 | 2,473 deaths", fc="#E8F6F3", ec="#117A65")
box(5.4, 4.4, 4.4, 1.4, "Temporal external validation cohort\nNHANES 2007-2008 (not used in any step\nof model building) n = 4,019 | 892 deaths", fc="#FDF2E9", ec="#CA6F1E")
arr(2.4, 4.4, 2.4, 3.6)
box(0.2, 1.7, 4.4, 1.9, "Model building (development data only)\n- 61 candidate predictors\n- Elastic-net Cox, 5-fold CV\n- Compact 12-variable Cox (age, sex forced)\n- Gradient boosting (ML comparator)", fc="#E8F6F3", ec="#117A65", fs=9)
arr(4.6, 2.6, 5.4, 2.6)
box(5.4, 1.7, 4.4, 1.9, "Frozen models applied to 2007-2008\n- Discrimination (C-index), paired ΔC\n- Calibration (slope, O:E, deciles)\n- NRI, IDI, decision curve analysis\n- Stratified performance", fc="#FDF2E9", ec="#CA6F1E", fs=9)
fig.tight_layout(); fig.savefig("figs/fig1_flow.png", dpi=300); plt.close()
# ---- Fig 2 discrimination
order = ["Age only", "Age + sex", "Original 9 variables (refit)", "Compact 12-variable Cox", "Elastic-net Cox (61 candidates)", "Gradient boosting"]
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=300)
for i, n in enumerate(order):
    m = R["models"][n]; ax.barh(i, m["c"], color=COL[n], height=0.62)
    ax.errorbar(m["c"], i, xerr=[[m["c"] - m["c_ci"][0]], [m["c_ci"][1] - m["c"]]], color="#222", capsize=3, lw=1.3)
    ax.text(m["c_ci"][1] + 0.004, i, f"{m['c']:.3f}", va="center", fontsize=9.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB.get(n, n) for n in order], fontsize=10); ax.invert_yaxis()
ax.set_xlim(0.70, 0.88); ax.set_xlabel("Harrell's C-index, 10-year mortality (95% CI)", fontsize=11)
ax.set_title("Discrimination in the temporal external validation cohort\n(NHANES 2007-2008, adults ≥ 40 years, n = 4,019)", fontsize=11.5); clean(ax); ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig("figs/fig2_discrimination.png", dpi=300); plt.close()
# ---- Fig 3 calibration
fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=300)
ax.plot([0, 80], [0, 80], ls="--", color="gray", lw=1.6, label="Ideal calibration")
for n, mk in (("Compact 12-variable Cox", "o"), ("Elastic-net Cox (61 candidates)", "s"), ("Age + sex", "^")):
    c = R["calibration"][n]; ax.plot([x["predicted"] * 100 for x in c], [x["observed"] * 100 for x in c], "-" + mk, color=COL[n], lw=2, ms=6.5, label=n)
ax.set_xlim(0, 80); ax.set_ylim(0, 80); ax.set_xlabel("Mean predicted 10-year risk (%)", fontsize=11.5); ax.set_ylabel("Observed 10-year mortality (%)", fontsize=11.5)
ax.set_title("Calibration by decile of predicted risk\n(NHANES 2007-2008 external validation, no recalibration)", fontsize=11.5); ax.legend(loc="upper left", fontsize=9.5); clean(ax)
fig.tight_layout(); fig.savefig("figs/fig3_calibration.png", dpi=300); plt.close()
# ---- Fig 4 decision curve
D = R["dca"]; th = np.array(D["threshold"]) * 100
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
ax.plot(th, D["treat_all"], color="#B03A2E", lw=1.6, ls=":", label="Treat all"); ax.axhline(0, color="#444", lw=1.4, ls="-", label="Treat none")
for n in ("Age + sex", "Original 9 variables (refit)", "Compact 12-variable Cox", "Elastic-net Cox (61 candidates)"):
    ax.plot(th, D[n], color=COL[n], lw=2.2 if "Compact" in n else 1.8, label=LAB.get(n, n))
ax.set_ylim(-0.02, 0.20); ax.set_xlabel("Threshold probability (%)", fontsize=11.5); ax.set_ylabel("Net benefit", fontsize=11.5)
ax.set_title("Decision curve analysis, 10-year mortality\n(NHANES 2007-2008; risks recalibrated-in-the-large)", fontsize=11.5); ax.legend(fontsize=9, loc="upper right"); clean(ax)
fig.tight_layout(); fig.savefig("figs/fig4_dca.png", dpi=300); plt.close()
print("ok")

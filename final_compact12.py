import sys, json
sys.argv = ["x", "40", "tmp_unused"]
src = open("pool_models.py").read()
exec(src[:src.index("t0 = time.time()\npaths")])
class PlainCox(Cox):
    def fit2(self, df, cols):
        self.prep = Prep().fit(df, cols); d2 = pd.DataFrame(self.prep.raw(df).values, columns=cols, index=df.index)
        d2["duration"] = df.duration.values; d2["event"] = df.event.values; self.fit(d2, cols); return self
    def lp(self, df): return (self.prep.raw(df).values - self.mean_) @ self.beta_
    def risk(self, df, t=H): return 1 - np.exp(-self.H0(t) * np.exp(self.lp(df)))
c2 = json.load(open("pool_out/compact2_results.json")); V12 = c2["12"]["vars"]; print(V12)
m12 = PlainCox().fit2(dev, V12)
t_e, e_e = ext.duration.values.astype(float), ext.event.values.astype(int)
P = pd.read_csv("pool_out/age40_ext_predictions.csv")
names = ["Age only", "Age + sex", "Original 9 variables (refit)", "Elastic-net Cox (61 candidates)", "Gradient boosting"]
risk = {n: P[f"risk_{i}"].values for i, n in enumerate(names)}
s12, r12 = m12.lp(ext), m12.risk(ext); risk["Compact 12-variable Cox"] = r12
rng = np.random.default_rng(7); idx = [rng.integers(0, len(ext), len(ext)) for _ in range(500)]
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
score = {n: risk[n] for n in risk}; score["Compact 12-variable Cox"] = s12
boot = {n: np.array([cindex(t_e[i], score[n][i], e_e[i]) for i in idx]) for n in score}
obs = km_risk(t_e, e_e); out = {"observed_10y": obs, "models": {}}
for n in score:
    c = cindex(t_e, score[n], e_e); d = dict(c=c, c_ci=pct(boot[n]), mean_pred=float(risk[n].mean()), oe=obs / risk[n].mean(), brier=float(np.mean((risk[n] - e_e) ** 2)))
    if n in ("Compact 12-variable Cox", "Elastic-net Cox (61 candidates)", "Original 9 variables (refit)", "Age + sex", "Age only"):
        lpx = s12 if n.startswith("Compact") else np.log(risk[n] / (1 - risk[n])) if False else None
    out["models"][n] = d
sl = calib_slope(s12, t_e, e_e); out["models"]["Compact 12-variable Cox"].update(slope=sl[0], slope_ci=[sl[1], sl[2]])
for ref in ("Age + sex", "Original 9 variables (refit)", "Elastic-net Cox (61 candidates)"):
    dd = boot["Compact 12-variable Cox"] - boot[ref]
    out["models"]["Compact 12-variable Cox"][f"delta_vs_{ref}"] = dict(delta=out["models"]["Compact 12-variable Cox"]["c"] - out["models"][ref]["c"], ci=pct(dd))
from scipy.optimize import brentq
lg = lambda r: np.log(np.clip(r, 1e-9, 1 - 1e-9) / (1 - np.clip(r, 1e-9, 1 - 1e-9)))
def rec(r):
    sh = brentq(lambda x: (1 / (1 + np.exp(-(lg(r) + x)))).mean() - obs, -6, 6); return 1 / (1 + np.exp(-(lg(r) + sh)))
rc = {n: rec(risk[n]) for n in risk}
def nri_idi(rn, ro, y, cuts=(0.10, 0.25)):
    cn, co = np.digitize(rn, cuts), np.digitize(ro, cuts); ev, ne = y == 1, y == 0; up, dn = cn > co, cn < co
    ne_ = up[ev].mean() - dn[ev].mean(); nn_ = dn[ne].mean() - up[ne].mean()
    ce = (rn[ev] > ro[ev]).mean() - (rn[ev] < ro[ev]).mean(); cn_ = (rn[ne] < ro[ne]).mean() - (rn[ne] > ro[ne]).mean()
    return dict(nri_cat=ne_ + nn_, nri_events=ne_, nri_nonevents=nn_, nri_cont=ce + cn_, idi=(rn[ev].mean() - ro[ev].mean()) - (rn[ne].mean() - ro[ne].mean()))
nri = {}
for kn, ko in (("Compact 12-variable Cox", "Age + sex"), ("Compact 12-variable Cox", "Original 9 variables (refit)"), ("Elastic-net Cox (61 candidates)", "Age + sex")):
    pt = nri_idi(rc[kn], rc[ko], e_e); bs = [nri_idi(rc[kn][i], rc[ko][i], e_e[i]) for i in idx[:400]]
    nri[f"{kn} vs {ko}"] = dict(point=pt, ci={k: pct([b[k] for b in bs]) for k in pt})
out["nri"] = nri
ths = np.round(np.arange(0.05, 0.501, 0.025), 3)
dca = lambda r: [float((((r >= p) & (e_e == 1)).sum() - ((r >= p) & (e_e == 0)).sum() * p / (1 - p)) / len(e_e)) for p in ths]
out["dca"] = dict(threshold=ths.tolist(), treat_all=[float(e_e.mean() - (1 - e_e.mean()) * p / (1 - p)) for p in ths], **{n: dca(rc[n]) for n in rc})
def dec(r):
    q = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False)
    return [dict(decile=g + 1, predicted=float(r[(q == g).values].mean()), observed=km_risk(t_e[(q == g).values], e_e[(q == g).values])) for g in range(10)]
out["calibration"] = {n: dec(risk[n]) for n in ("Compact 12-variable Cox", "Elastic-net Cox (61 candidates)", "Age + sex")}
strata = {"Age 40-59": ext.RIDAGEYR < 60, "Age >= 60": ext.RIDAGEYR >= 60, "Women": ext.male == 0, "Men": ext.male == 1}
out["stratified"] = {k: dict(n=int(m.sum()), events=int(e_e[m.values].sum()), **{n: cindex(t_e[m.values], score[n][m.values], e_e[m.values]) for n in score}) for k, m in strata.items()}
# risk groups for the compact model (applied with fixed dev-derived cutoffs 10% / 25%)
g = np.digitize(r12, (0.10, 0.25)); out["risk_groups"] = [dict(group=["<10%", "10-25%", ">=25%"][i], n=int((g == i).sum()), observed=float(e_e[g == i].mean()), predicted=float(r12[g == i].mean())) for i in range(3)]
# coefficients with CI
out["compact_hr"] = c2["12"]["hr"]; out["compact_sizes"] = {K: dict(n=c2[K]["n_vars"], c=c2[K]["c"], ci=c2[K]["c_ci"], slope=c2[K]["slope"], oe=c2[K]["oe"]) for K in ("8", "12", "16", "20")}
# baseline hazard so the model can be applied by others
out["model_card"] = dict(vars=V12, beta_per_unit=m12.beta_.tolist(), means=m12.mean_.tolist(), H0_120=float(m12.H0(120)), prep_median=m12.prep.med.to_dict(), prep_lo=m12.prep.lo.to_dict(), prep_hi=m12.prep.hi.to_dict())
pd.DataFrame({"SEQN": ext.SEQN, "duration": t_e, "event": e_e, "risk_compact12": r12, "lp_compact12": s12}).to_csv("pool_out/compact12_ext_predictions.csv", index=False)
json.dump(out, open("pool_out/final_results.json", "w"), indent=1, default=float)
for n, d in out["models"].items(): print(n, round(d["c"], 4), [round(x, 4) for x in d["c_ci"]], round(d["oe"], 3))
print(out["risk_groups"]); print({k: {a: round(b, 3) for a, b in v["point"].items()} for k, v in nri.items()})
print(out["models"]["Compact 12-variable Cox"])

"""Multi-cycle NHANES 2001-2008 mortality models. usage: python3 -I pool_models.py <min_age> <out_prefix>
Develop on 2001-2006 (cycles B,C,D), temporal external validation on 2007-2008 (E)."""
import sys, json, time, warnings
sys.path.insert(0, ".")
from cox_lib import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
MIN_AGE, PFX = int(sys.argv[1]), sys.argv[2]
SEED = 42; B_BOOT = 500
ORIG9 = ["RIDAGEYR", "male", "SBP_mean", "abdominal_obesity", "current_daily_smoker", "edu_more_than_hs", "LBXWBCSI", "LBXHGB", "BPXPLS"]
CAND = ["RIDAGEYR", "male", "race_black", "race_mexican", "race_other", "edu_more_than_hs", "marital_never", "marital_widowed", "INDFMPIR",
        "current_daily_smoker", "current_smoker_any", "former_smoker", "SBP_mean", "DBP_mean", "BPXPLS", "BMXBMI", "BMXWAIST", "BMXHT",
        "abdominal_obesity", "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI",
        "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMPSI", "nlr", "LBXSAL", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSBU", "LBXSCA",
        "LBXSC3SI", "log_gtt", "LBXSGL", "LBXSIR", "LBXSLDSI", "LBXSPH", "LBXSTB", "LBXSTP", "log_trig", "LBXSUA", "log_creat",
        "LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBXSOSSI", "LBXSGB", "log_crp", "LBXGH", "diabetes_hba1c", "LBXTC", "LBDHDL"]
d = pd.read_csv("pool_analytic.csv"); d = d[d.RIDAGEYR >= MIN_AGE].reset_index(drop=True)
dev = d[d.cycle != "2007-2008"].reset_index(drop=True); ext = d[d.cycle == "2007-2008"].reset_index(drop=True)
print(f"min_age={MIN_AGE} dev n={len(dev)} ev={dev.event.sum()} | ext n={len(ext)} ev={ext.event.sum()}", flush=True)

class Prep:
    """winsorise (0.5/99.5 pct of train), median-impute, standardise; all fitted on training rows only."""
    def fit(self, df, cols):
        self.cols = cols; x = df[cols].astype(float)
        self.lo = x.quantile(0.005); self.hi = x.quantile(0.995)
        x = x.clip(self.lo, self.hi, axis=1); self.med = x.median(); x = x.fillna(self.med)
        self.mu = x.mean(); self.sd = x.std(ddof=1).replace(0, 1); return self
    def raw(self, df):
        return df[self.cols].astype(float).clip(self.lo, self.hi, axis=1).fillna(self.med)
    def Z(self, df): return ((self.raw(df) - self.mu) / self.sd).values

def soft(x, a): return np.sign(x) * np.maximum(np.abs(x) - a, 0)
def fista(Z, t, e, lam, alpha, b0, maxit=250, tol=2e-5):
    l2, l1 = lam * (1 - alpha), lam * alpha
    f = lambda b: neg_ll_grad(b, Z, t, e, l2)
    b = b0.copy(); y = b.copy(); tk = 1.0; L = 1.0; fy, gy = f(y)
    for _ in range(maxit):
        while True:
            bn = soft(y - gy / L, l1 / L); fn = f(bn)[0]; dd = bn - y
            if fn <= fy + gy @ dd + 0.5 * L * dd @ dd + 1e-12: break
            L *= 2
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        conv = np.max(np.abs(bn - b)) < tol
        y = bn + ((tk - 1) / tn) * (bn - b); b = bn; tk = tn
        if conv: break
        fy, gy = f(y)
    return b

def cv_path(alpha, lams_frac, nl=20, K=5):
    X = dev; t = X.duration.values.astype(float); e = X.event.values.astype(int)
    pr = Prep().fit(X, CAND); Z0 = pr.Z(X)
    lam_max = np.max(np.abs(neg_ll_grad(np.zeros(Z0.shape[1]), Z0, t, e, 0)[1])) / max(alpha, 1e-3)
    lams = lam_max * np.logspace(0, -2.5, nl)
    folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED).split(X, e))
    loss = np.zeros((K, nl)); nnz = np.zeros(nl)
    for k, (a, b) in enumerate(folds):
        p = Prep().fit(X.iloc[a], CAND); Za, Zb = p.Z(X.iloc[a]), p.Z(X.iloc[b])
        ta, ea, tb, eb = t[a], e[a], t[b], e[b]; beta = np.zeros(Za.shape[1])
        for i, lam in enumerate(lams):
            beta = fista(Za, ta, ea, lam, alpha, beta); loss[k, i] = neg_ll_grad(beta, Zb, tb, eb, 0)[0]
    # full-data path for df
    beta = np.zeros(Z0.shape[1])
    for i, lam in enumerate(lams):
        beta = fista(Z0, t, e, lam, alpha, beta); nnz[i] = (np.abs(beta) > 1e-8).sum()
    m = loss.mean(0); se = loss.std(0, ddof=1) / np.sqrt(K)
    return dict(alpha=alpha, lams=lams, cv=m, se=se, nnz=nnz)

t0 = time.time()
paths = Parallel(n_jobs=2)(delayed(cv_path)(a, None) for a in (1.0, 0.5))
print("CV paths done", round(time.time() - t0), "s", flush=True)
best = min(paths, key=lambda p: p["cv"].min())
imin = int(best["cv"].argmin()); thr = best["cv"][imin] + best["se"][imin]
i1se = int(np.where(best["cv"] <= thr)[0][0])   # largest lambda within 1 SE
ALPHA, LAM = best["alpha"], float(best["lams"][i1se])
print(f"chosen alpha={ALPHA} lambda(1se)={LAM:.5f} nnz={best['nnz'][i1se]:.0f} (min: lam={best['lams'][imin]:.5f} nnz={best['nnz'][imin]:.0f})", flush=True)

class PenCox(Cox):
    def fit_pen(self, df, cols, lam, alpha):
        self.cols = list(cols); self.prep = Prep().fit(df, self.cols); Z = self.prep.Z(df)
        t = df.duration.values.astype(float); e = df.event.values.astype(int)
        b = fista(Z, t, e, lam, alpha, np.zeros(Z.shape[1]), maxit=1500, tol=1e-9)
        self.beta_std_ = b; self.sd_ = self.prep.sd.values; self.beta_ = b / self.sd_
        self.mean_ = self.prep.mu.values; self.std_ = self.sd_
        lp = Z @ b; self._baseline(t, e, lp - lp.mean()); self._lpc = lp.mean(); return self
    def lp(self, df): return self.prep.Z(df) @ self.beta_std_ - self._lpc

class PlainCox(Cox):  # unpenalised-ish Cox with winsor/impute for low-dim reference models
    def fit2(self, df, cols):
        self.prep = Prep().fit(df, cols); d2 = pd.DataFrame(self.prep.raw(df).values, columns=cols, index=df.index)
        d2["duration"] = df.duration.values; d2["event"] = df.event.values
        self.fit(d2, cols); return self
    def lp(self, df): return (self.prep.raw(df).values - self.mean_) @ self.beta_
    def risk(self, df, t=H): return 1 - np.exp(-self.H0(t) * np.exp(self.lp(df)))

def build_models():
    M = {}
    M["Age only"] = PlainCox().fit2(dev, ["RIDAGEYR"])
    M["Age + sex"] = PlainCox().fit2(dev, ["RIDAGEYR", "male"])
    M["Original 9 variables (refit)"] = PlainCox().fit2(dev, ORIG9)
    M[f"Penalised Cox (elastic net, {len(CAND)} candidates)"] = PenCox().fit_pen(dev, CAND, LAM, ALPHA)
    return M
models = build_models(); NAME_PEN = [k for k in models if k.startswith("Penalised")][0]
pen = models[NAME_PEN]
sel = [(c, float(b), float(bs)) for c, b, bs in zip(pen.cols, pen.beta_, pen.beta_std_) if abs(bs) > 1e-8]
sel.sort(key=lambda x: -abs(x[2]))
print("selected", len(sel), [s[0] for s in sel], flush=True)

# --- ML comparator: gradient boosting classifier for 10-year death (all participants have complete 10-y status)
grid = [(lr, leaf, l2) for lr in (0.03, 0.06) for leaf in (4, 8) for l2 in (1.0, 10.0)]
Xd = dev[CAND].astype(float).values; yd = dev.event.values
cvk = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd)); gs = []
for lr, leaf, l2 in grid:
    cs = []
    for a, b in cvk:
        g = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaf, l2_regularization=l2, min_samples_leaf=50, max_iter=600,
                                           early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=SEED).fit(Xd[a], yd[a])
        cs.append(cindex(dev.duration.values[b], g.predict_proba(Xd[b])[:, 1], dev.event.values[b]))
    gs.append(np.mean(cs))
bg = grid[int(np.argmax(gs))]; print("GBM best", bg, round(max(gs), 4), flush=True)
gbm = HistGradientBoostingClassifier(learning_rate=bg[0], max_leaf_nodes=bg[1], l2_regularization=bg[2], min_samples_leaf=50, max_iter=600,
                                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=SEED).fit(Xd, yd)

# --- external evaluation
t_e, e_e = ext.duration.values.astype(float), ext.event.values.astype(int)
scores, risk = {}, {}
for k, m in models.items():
    scores[k] = m.lp(ext); risk[k] = m.risk(ext) if hasattr(m, "risk") else None
pg = gbm.predict_proba(ext[CAND].astype(float).values)[:, 1]
scores["Gradient boosting (ML)"] = np.log(pg / (1 - pg)); risk["Gradient boosting (ML)"] = pg
rng = np.random.default_rng(7); idx = [rng.integers(0, len(ext), len(ext)) for _ in range(B_BOOT)]
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
boot = {k: np.array([cindex(t_e[i], s[i], e_e[i]) for i in idx]) for k, s in scores.items()}
obs10 = km_risk(t_e, e_e)
res = {}
for k, s in scores.items():
    c = cindex(t_e, s, e_e); sl = calib_slope(s, t_e, e_e)
    r = risk[k]
    res[k] = dict(c=c, c_ci=pct(boot[k]), slope=sl[0], slope_ci=[sl[1], sl[2]], oe=obs10 / r.mean(), mean_pred=float(r.mean()),
                  brier=float(np.mean((r - e_e) ** 2)))
for k in scores:
    for ref in ("Age only", "Age + sex"):
        if k != ref:
            dd = boot[k] - boot[ref]; res[k][f"delta_vs_{ref}"] = dict(delta=res[k]["c"] - res[ref]["c"], ci=pct(dd))
res["_observed_10y_risk"] = obs10
# internal (development) CV/optimism for context
print({k: round(v["c"], 4) for k, v in res.items() if isinstance(v, dict)}, flush=True)

# --- recalibration-in-the-large on external (equal treatment of all models), used for NRI / DCA
def shift_to(r_raw, target):
    from scipy.optimize import brentq
    lo = lambda r: np.log(r / (1 - r))
    f = lambda dlt: (1 / (1 + np.exp(-(lo(np.clip(r_raw, 1e-9, 1 - 1e-9)) + dlt)))).mean() - target
    return brentq(f, -6, 6)
rec = {k: 1 / (1 + np.exp(-(np.log(np.clip(r, 1e-9, 1 - 1e-9) / (1 - np.clip(r, 1e-9, 1 - 1e-9))) + shift_to(r, obs10)))) for k, r in risk.items()}

def nri_idi(r_new, r_old, y, cuts=(0.10, 0.25)):
    cat = lambda r: np.digitize(r, cuts)
    cn, co = cat(r_new), cat(r_old); ev, ne = y == 1, y == 0
    up, dn = cn > co, cn < co
    nri_e = (up[ev].mean() - dn[ev].mean()); nri_n = (dn[ne].mean() - up[ne].mean())
    cont_e = (r_new[ev] > r_old[ev]).mean() - (r_new[ev] < r_old[ev]).mean()
    cont_n = (r_new[ne] < r_old[ne]).mean() - (r_new[ne] > r_old[ne]).mean()
    idi = (r_new[ev].mean() - r_old[ev].mean()) - (r_new[ne].mean() - r_old[ne].mean())
    return dict(nri_cat=nri_e + nri_n, nri_cat_events=nri_e, nri_cat_nonevents=nri_n, nri_cont=cont_e + cont_n, idi=idi)
def nri_boot(kn, ko):
    out = [nri_idi(rec[kn][i], rec[ko][i], e_e[i]) for i in idx[:300]]
    return {kk: pct([o[kk] for o in out]) for kk in out[0]}
nri = {}
for kn in [NAME_PEN, "Gradient boosting (ML)", "Original 9 variables (refit)"]:
    for ko in ["Age + sex"] + ([] if kn == "Original 9 variables (refit)" else ["Original 9 variables (refit)"]):
        nri[f"{kn} vs {ko}"] = dict(point=nri_idi(rec[kn], rec[ko], e_e), ci=nri_boot(kn, ko))
def dca(r, y, ths):
    n = len(y); return [float((((r >= p) & (y == 1)).sum() - ((r >= p) & (y == 0)).sum() * p / (1 - p)) / n) for p in ths]
ths = np.round(np.arange(0.05, 0.51, 0.025), 3)
dc = {"threshold": ths.tolist(), "treat_all": [float(e_e.mean() - (1 - e_e.mean()) * p / (1 - p)) for p in ths], "treat_none": [0.0] * len(ths)}
for k in rec: dc[k] = dca(rec[k], e_e, ths)

# --- calibration by decile (raw predicted risks, penalised Cox + original 9)
def deciles(r, key):
    q = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False); rows = []
    for g in range(10):
        m = (q == g).values; rows.append(dict(decile=g + 1, n=int(m.sum()), predicted=float(r[m].mean()), observed=km_risk(t_e[m], e_e[m])))
    return rows
cal = {k: deciles(risk[k], k) for k in (NAME_PEN, "Original 9 variables (refit)", "Gradient boosting (ML)")}

# --- stratified discrimination
strata = {"Age 40-59": ext.RIDAGEYR < 60, "Age >= 60": ext.RIDAGEYR >= 60, "Women": ext.male == 0, "Men": ext.male == 1}
if MIN_AGE < 40: strata = {"Age 18-39": ext.RIDAGEYR < 40, **strata}
strat = {}
for nm, m in strata.items():
    m = m.values; strat[nm] = dict(n=int(m.sum()), events=int(e_e[m].sum()),
        **{k: cindex(t_e[m], scores[k][m], e_e[m]) for k in scores})

# --- internal-external cross-validation by cycle (train on 3 cycles incl. dev+ext pool, test on the 4th)
iecv = {}
for cyc in sorted(d.cycle.unique()):
    tr_, te_ = d[d.cycle != cyc].reset_index(drop=True), d[d.cycle == cyc].reset_index(drop=True)
    row = {}
    for nm, cols in (("Age only", ["RIDAGEYR"]), ("Age + sex", ["RIDAGEYR", "male"]), ("Original 9 variables", ORIG9)):
        m = PlainCox().fit2(tr_, cols); row[nm] = cindex(te_.duration, m.lp(te_), te_.event)
    p_ = PenCox().fit_pen(tr_, CAND, LAM, ALPHA); row["Penalised Cox"] = cindex(te_.duration, p_.lp(te_), te_.event)
    row["n"] = len(te_); row["events"] = int(te_.event.sum()); iecv[cyc] = row; print("IECV", cyc, row, flush=True)

out = dict(min_age=MIN_AGE, n_dev=len(dev), ev_dev=int(dev.event.sum()), n_ext=len(ext), ev_ext=int(ext.event.sum()),
           dev_by_cycle=dev.groupby("cycle").agg(n=("event", "size"), ev=("event", "sum")).to_dict("index"),
           alpha=ALPHA, lam_1se=LAM, nnz_1se=int(best["nnz"][i1se]), lam_min=float(best["lams"][imin]), nnz_min=int(best["nnz"][imin]),
           cv_paths=[dict(alpha=p["alpha"], lams=p["lams"].tolist(), cv=p["cv"].tolist(), se=p["se"].tolist(), nnz=p["nnz"].tolist()) for p in paths],
           selected=[dict(var=c, coef_per_unit=b, coef_per_sd=bs, hr_per_sd=float(np.exp(bs))) for c, b, bs in sel],
           gbm_params=dict(learning_rate=bg[0], max_leaf_nodes=bg[1], l2=bg[2], cv_c=float(max(gs)), n_iter=int(gbm.n_iter_)),
           external=res, nri=nri, dca=dc, calibration_deciles=cal, stratified=strat, iecv=iecv,
           shift_to_recalibrate={k: float(shift_to(r, obs10)) for k, r in risk.items()})
json.dump(out, open(f"pool_out/{PFX}_results.json", "w"), indent=1, default=float)
pd.DataFrame({"SEQN": ext.SEQN, "duration": t_e, "event": e_e, **{f"risk_{i}": risk[k] for i, k in enumerate(risk)}}).to_csv(f"pool_out/{PFX}_ext_predictions.csv", index=False)
print("DONE", round(time.time() - t0), "s", flush=True)

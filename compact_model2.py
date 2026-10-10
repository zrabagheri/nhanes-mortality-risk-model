"""Compact (parsimonious) model chosen from the LASSO path on development data only; refit unpenalised; validate on 2007-2008."""
import sys, json
sys.argv = ["x", "40", "tmp_unused"]
src = open("pool_models.py").read()
head = src[:src.index("t0 = time.time()\npaths")]          # imports, data, Prep, fista, cv_path defs
exec(head)
from scipy.stats import norm
class PlainCox(Cox):
    def fit2(self, df, cols):
        self.prep = Prep().fit(df, cols); d2 = pd.DataFrame(self.prep.raw(df).values, columns=cols, index=df.index)
        d2["duration"] = df.duration.values; d2["event"] = df.event.values; self.fit(d2, cols); return self
    def lp(self, df): return (self.prep.raw(df).values - self.mean_) @ self.beta_
    def risk(self, df, t=H): return 1 - np.exp(-self.H0(t) * np.exp(self.lp(df)))
def wald(m, df):
    d2 = pd.DataFrame(m.prep.raw(df).values, columns=m.cols, index=df.index)
    Z = (d2.values - m.mean_) / m.std_; t = df.duration.values.astype(float); e = df.event.values.astype(int)
    b = m.beta_std_; p = len(b); eps = 1e-4; Hm = np.zeros((p, p))
    g0 = lambda x: neg_ll_grad(x, Z, t, e, m.pen)[1]
    for j in range(p):
        bp = b.copy(); bp[j] += eps; bm = b.copy(); bm[j] -= eps; Hm[:, j] = (g0(bp) - g0(bm)) / (2 * eps)
    cov = np.linalg.inv(Hm * len(df)); se = np.sqrt(np.diag(cov)) / m.std_
    return se
pr = Prep().fit(dev, CAND); Z0 = pr.Z(dev); t = dev.duration.values.astype(float); e = dev.event.values.astype(int)
lam_max = np.max(np.abs(neg_ll_grad(np.zeros(Z0.shape[1]), Z0, t, e, 0)[1]))
beta = np.zeros(Z0.shape[1]); path = []
for lam in lam_max * np.logspace(0, -2.2, 40):
    beta = fista(Z0, t, e, lam, 1.0, beta, maxit=400, tol=1e-6); path.append(((np.abs(beta) > 1e-8).sum(), [CAND[i] for i in np.where(np.abs(beta) > 1e-8)[0]]))
res = {}; t_e, e_e = ext.duration.values.astype(float), ext.event.values.astype(int)
rng = np.random.default_rng(7); idx = [rng.integers(0, len(ext), len(ext)) for _ in range(500)]
pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
pj = json.load(open("pool_out/age40_results.json")); c_full = pj["external"][[k for k in pj["external"] if k.startswith("Penalised")][0]]["c"]
full_pred = pd.read_csv("pool_out/age40_ext_predictions.csv")
entry = []
for n, vs in path:
    for v in vs:
        if v not in entry: entry.append(v)
print("entry order:", entry[:25], flush=True)
for K in (8, 12, 16, 20):
    vars_ = ["RIDAGEYR", "male"] + [v for v in entry if v not in ("RIDAGEYR", "male")][:K - 2]
    m = PlainCox().fit2(dev, vars_); s = m.lp(ext); r = m.risk(ext)
    boot = np.array([cindex(t_e[i], s[i], e_e[i]) for i in idx]); c = cindex(t_e, s, e_e); sl = calib_slope(s, t_e, e_e)
    se = wald(m, dev)
    res[K] = dict(n_vars=len(vars_), vars=vars_, c=c, c_ci=pct(boot), slope=sl[0], slope_ci=[sl[1], sl[2]], oe=km_risk(t_e, e_e) / r.mean(),
                  brier=float(np.mean((r - e_e) ** 2)),
                  hr=[dict(var=v, hr_per_sd=float(np.exp(b * sd)), hr_per_unit=float(np.exp(b)), lo_unit=float(np.exp(b - 1.96 * q)), hi_unit=float(np.exp(b + 1.96 * q)),
                           p=float(2 * (1 - norm.cdf(abs(b / q))))) for v, b, sd, q in zip(m.cols, m.beta_, m.std_, se)])
    print(K, len(vars_), round(c, 4), round(sl[0], 3), flush=True)
    if K == 12:
        np.save("pool_out/compact12b_risk.npy", r); np.save("pool_out/compact12b_lp.npy", s)
        q = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False)
        res["deciles12"] = [dict(decile=g + 1, predicted=float(r[(q == g).values].mean()), observed=km_risk(t_e[(q == g).values], e_e[(q == g).values])) for g in range(10)]
json.dump(res, open("pool_out/compact2_results.json", "w"), indent=1, default=float)
print("full-model C", c_full)

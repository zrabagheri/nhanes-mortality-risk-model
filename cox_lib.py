import numpy as np, pandas as pd
from scipy.optimize import minimize
H=120
PEN=0.01
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer

SEED = 42
H = 120
PEN = 0.01

# ------------------------------------------------------------------ Cox PH (Efron) --------------------
def _prep(time, event):
    order = np.argsort(time, kind="mergesort")
    return order

def neg_ll_grad(beta, X, time, event, pen):
    """Mean negative Efron partial log-likelihood + 0.5*pen*||beta||^2, with gradient."""
    n, p = X.shape
    order = np.argsort(time, kind="mergesort")
    X, time, event = X[order], time[order], event[order]
    lp = X @ beta
    lp -= lp.max()
    w = np.exp(lp)
    # reverse cumulative sums -> risk-set sums for every row (risk set = time >= t_i)
    S0_rev = np.cumsum(w[::-1])[::-1]
    S1_rev = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]
    ll = 0.0
    grad = np.zeros(p)
    ev_idx = np.where(event == 1)[0]
    ev_times = time[ev_idx]
    for t in np.unique(ev_times):
        D = ev_idx[ev_times == t]
        d = len(D)
        first = np.searchsorted(time, t, side="left")
        s0, s1 = S0_rev[first], S1_rev[first]
        tw = w[D].sum()
        tx = (w[D][:, None] * X[D]).sum(axis=0)
        ll += lp[D].sum()
        grad += X[D].sum(axis=0)
        fr = np.arange(d) / d
        den = s0 - fr * tw
        ll -= np.log(den).sum()
        grad -= ((s1[None, :] - fr[:, None] * tx[None, :]) / den[:, None]).sum(axis=0)
    f = -ll / n + 0.5 * pen * np.sum(beta ** 2)
    g = -grad / n + pen * beta
    return f, g

class Cox:
    def __init__(self, penalizer=PEN):
        self.pen = penalizer

    def fit(self, df, cols):
        self.cols = list(cols)
        X = df[self.cols].values.astype(float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        Z = (X - self.mean_) / self.std_
        r = minimize(neg_ll_grad, np.zeros(Z.shape[1]),
                     args=(Z, df["duration"].values.astype(float), df["event"].values.astype(int), self.pen),
                     jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        self.beta_std_ = r.x
        self.beta_ = r.x / self.std_
        self.converged_ = r.success
        # Breslow baseline cumulative hazard with centred covariates
        lp = (X - self.mean_) @ self.beta_
        self._baseline(df["duration"].values.astype(float), df["event"].values.astype(int), lp)
        return self

    def _baseline(self, time, event, lp):
        order = np.argsort(time, kind="mergesort")
        time, event, lp = time[order], event[order], lp[order]
        w = np.exp(lp)
        S0 = np.cumsum(w[::-1])[::-1]
        ts = np.unique(time[event == 1])
        h = []
        for t in ts:
            d = ((time == t) & (event == 1)).sum()
            first = np.searchsorted(time, t, side="left")
            h.append(d / S0[first])
        self.base_t_ = ts
        self.base_H_ = np.cumsum(h)

    def lp(self, df):
        return (df[self.cols].values.astype(float) - self.mean_) @ self.beta_

    def H0(self, t):
        i = np.searchsorted(self.base_t_, t, side="right") - 1
        return 0.0 if i < 0 else self.base_H_[i]

    def risk(self, df, t=H):
        return 1 - np.exp(-self.H0(t) * np.exp(self.lp(df)))

def cindex(time, score, event):
    """Harrell's C; score = higher means higher risk. Ties in score count 0.5."""
    time = np.asarray(time, float); score = np.asarray(score, float); event = np.asarray(event, int)
    num = den = 0.0
    for i in np.where(event == 1)[0]:
        m = time > time[i]
        if not m.any():
            continue
        den += m.sum()
        num += (score[i] > score[m]).sum() + 0.5 * (score[i] == score[m]).sum()
    return num / den

# ------------------------------------------------------------------ Km helper ------------------------
def km_risk(time, event, t=H):
    time = np.asarray(time, float); event = np.asarray(event, int)
    s = 1.0
    for u in np.unique(time[(event == 1) & (time <= t)]):
        at_risk = (time >= u).sum()
        d = ((time == u) & (event == 1)).sum()
        s *= 1 - d / at_risk
    return 1 - s

def calib_slope(lp, time, event):
    """Cox regression of outcome on the linear predictor (single covariate, no penalty)."""
    d = pd.DataFrame({"lp": lp, "duration": time, "event": event})
    m = Cox(penalizer=0.0).fit(d, ["lp"])
    # SE via numerical Hessian on the standardised scale
    z = (d["lp"].values - m.mean_[0]) / m.std_[0]
    b = m.beta_std_
    eps = 1e-4
    f = lambda x: neg_ll_grad(np.array([x]), z[:, None], d["duration"].values.astype(float), d["event"].values.astype(int), 0.0)[0]
    h = (f(b[0] + eps) - 2 * f(b[0]) + f(b[0] - eps)) / eps ** 2
    se = np.sqrt(1 / (h * len(d))) / m.std_[0]
    return m.beta_[0], m.beta_[0] - 1.96 * se, m.beta_[0] + 1.96 * se


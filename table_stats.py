import json, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
d = pd.read_csv("pool_analytic.csv"); a = d[d.RIDAGEYR >= 40]
dev = a[a.cycle != "2007-2008"]; ext = a[a.cycle == "2007-2008"]
def ms(x): x = x.dropna(); return f"{x.mean():.1f} ({x.std():.1f})"
def mi(x): x = x.dropna(); return f"{x.median():.2f} ({x.quantile(.25):.2f}-{x.quantile(.75):.2f})"
def pc(x, v=1): x = x.dropna(); return f"{int((x==v).sum()):,} ({100*(x==v).mean():.1f}%)"
rows = []
def row(lbl, f):
    rows.append([lbl, f(dev), f(ext), f(a)])
row("Participants, n", lambda x: f"{len(x):,}")
row("Deaths within 10 years, n (%)", lambda x: pc(x.event))
row("Age, years, mean (SD)", lambda x: ms(x.RIDAGEYR))
row("Male sex, n (%)", lambda x: pc(x.male))
row("Non-Hispanic Black, n (%)", lambda x: pc(x.race_black))
row("Mexican American, n (%)", lambda x: pc(x.race_mexican))
row("More than high school education, n (%)", lambda x: pc(x.edu_more_than_hs))
row("Income-to-poverty ratio, mean (SD)", lambda x: ms(x.INDFMPIR))
row("Current smoker (daily or some days), n (%)", lambda x: pc(x.current_smoker_any))
row("Former smoker, n (%)", lambda x: pc(x.former_smoker))
row("Body mass index, kg/m², mean (SD)", lambda x: ms(x.BMXBMI))
row("Systolic blood pressure, mmHg, mean (SD)", lambda x: ms(x.SBP_mean))
row("Pulse, beats/min, mean (SD)", lambda x: ms(x.BPXPLS))
row("Hemoglobin A1c ≥ 6.5%, n (%)", lambda x: pc(x.diabetes_hba1c))
row("Serum creatinine, mg/dL, median (IQR)", lambda x: mi(x.LBXSCR if "LBXSCR" in x else np.exp(x.log_creat)))
row("Serum albumin, g/dL, mean (SD)", lambda x: ms(x.LBXSAL))
row("Red cell distribution width, %, mean (SD)", lambda x: ms(x.LBXRDW))
row("Neutrophil-to-lymphocyte ratio, median (IQR)", lambda x: mi(x.nlr))
row("GGT, U/L, median (IQR)", lambda x: mi(np.exp(x.log_gtt)))
row("C-reactive protein, mg/dL, median (IQR)", lambda x: mi(np.exp(x.log_crp)))
miss = (dev[[c for c in json.load(open("pool_out/age40_results.json"))["selected"][0:0]]].isna().mean() if False else None)
cand = ["RIDAGEYR", "male", "race_black", "race_mexican", "race_other", "edu_more_than_hs", "marital_never", "marital_widowed", "INDFMPIR", "current_daily_smoker", "current_smoker_any", "former_smoker", "SBP_mean", "DBP_mean", "BPXPLS", "BMXBMI", "BMXWAIST", "BMXHT", "abdominal_obesity", "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMPSI", "nlr", "LBXSAL", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSBU", "LBXSCA", "LBXSC3SI", "log_gtt", "LBXSGL", "LBXSIR", "LBXSLDSI", "LBXSPH", "LBXSTB", "LBXSTP", "log_trig", "LBXSUA", "log_creat", "LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBXSOSSI", "LBXSGB", "log_crp", "LBXGH", "diabetes_hba1c", "LBXTC", "LBDHDL"]
ms_ = {c: float(100 * dev[c].isna().mean()) for c in cand}
print("max missing dev", max(ms_.values()), "n>10%", sum(v > 10 for v in ms_.values()))
json.dump(dict(rows=rows, missing=ms_, n_cand=len(cand)), open("pool_out/table_stats.json", "w"), indent=1)
for r in rows: print(r)

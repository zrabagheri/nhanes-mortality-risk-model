"""Harmonise NHANES 2001-2008 (cycles B-E) + linked mortality (2019) into one analytic file.
usage: python3 -I build_pool.py pool_dir out_csv"""
import sys, numpy as np, pandas as pd
P, OUT = sys.argv[1], sys.argv[2]
H = 120
CYC = {"B": "2001-2002", "C": "2003-2004", "D": "2005-2006", "E": "2007-2008"}
BIO = {"B": "L40_B", "C": "L40_C", "D": "BIOPRO_D", "E": "BIOPRO_E"}
CBC = {"B": "L25_B", "C": "L25_C", "D": "CBC_D", "E": "CBC_E"}
CRP = {"B": "L11_B", "C": "L11_C", "D": "CRP_D", "E": "CRP_E"}
GHB = {"B": "L10_B", "C": "L10_C", "D": "GHB_D", "E": "GHB_E"}
TCH = {"B": "L13_B", "C": "L13_C", "D": "TCHOL_D", "E": "TCHOL_E"}
HDL = {"B": "L13_B", "C": "L13_C", "D": "HDL_D", "E": "HDL_E"}
rd = lambda n: pd.read_sas(f"{P}/{n}.xpt", format="xport")
ren_bio = {"LBDSAPSI": "LBXSAPSI", "LBDSLDSI": "LBXSLDSI", "LBDSPH": "LBXSPH", "LBDSTB": "LBXSTB", "LBDSCR": "LBXSCR"}
bio_keep = ["LBXSAL", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSBU", "LBXSCA", "LBXSCH", "LBXSC3SI", "LBXSGTSI", "LBXSGL",
            "LBXSIR", "LBXSLDSI", "LBXSPH", "LBXSTB", "LBXSTP", "LBXSTR", "LBXSUA", "LBXSCR", "LBXSNASI", "LBXSKSI",
            "LBXSCLSI", "LBXSOSSI", "LBXSGB"]
cbc_keep = ["LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI",
            "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMPSI"]
out = []
for s, lab in CYC.items():
    demo = rd(f"DEMO_{s}"); bpx = rd(f"BPX_{s}"); bmx = rd(f"BMX_{s}"); smq = rd(f"SMQ_{s}")
    bio = rd(BIO[s]).rename(columns=ren_bio)[["SEQN"] + bio_keep]
    cbc = rd(CBC[s])[["SEQN"] + cbc_keep]
    crp = rd(CRP[s])[["SEQN", "LBXCRP"]]
    ghb = rd(GHB[s])[["SEQN", "LBXGH"]]
    tch = rd(TCH[s])[["SEQN", "LBXTC"]]
    h = rd(HDL[s]); hcol = [c for c in ("LBDHDL", "LBXHDD", "LBDHDD") if c in h][0]
    h = h[["SEQN", hcol]].rename(columns={hcol: "LBDHDL"})
    mort = pd.read_fwf(f"{P}/NHANES_{lab.replace('-', '_')}_MORT_2019_PUBLIC.dat", colspecs=[(0, 5), (14, 15), (15, 16), (42, 45)],
                       names=["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT"])
    mort = mort.replace({"": pd.NA, ".": pd.NA, "..": pd.NA}).apply(pd.to_numeric, errors="coerce")
    me = mort[mort.ELIGSTAT == 1]
    df = me[["SEQN", "MORTSTAT", "PERMTH_INT"]].copy()
    dcols = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "INDFMPIR", "WTMEC2YR"]
    df = df.merge(demo[dcols], on="SEQN", how="left")
    bcols = ["SEQN", "BPXPLS"] + [c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4") if c in bpx]
    df = df.merge(bpx[bcols], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT", "BMXWT"]], on="SEQN", how="left")
    df = df.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
    for t in (bio, cbc, crp, ghb, tch, h):
        df = df.merge(t, on="SEQN", how="left")
    df["cycle"] = lab
    out.append(df)
df = pd.concat(out, ignore_index=True)
n0 = len(df)
df = df.dropna(subset=["MORTSTAT", "PERMTH_INT"]); df = df[df.PERMTH_INT > 0]
unk = (df.MORTSTAT == 0) & (df.PERMTH_INT < H)
print("mortality-eligible:", n0, "| excluded alive <120 mo:", int(unk.sum()))
df = df[~unk].copy()
df["event"] = ((df.MORTSTAT == 1) & (df.PERMTH_INT <= H)).astype(int)
df["duration"] = np.minimum(df.PERMTH_INT, H)
df["male"] = (df.RIAGENDR == 1).astype(int)
sy = [c for c in df if c.startswith("BPXSY")]; di = [c for c in df if c.startswith("BPXDI")]
df["SBP_mean"] = df[sy].mean(axis=1); df["DBP_mean"] = df[di].mean(axis=1)
df["abdominal_obesity"] = np.where(df.BMXWAIST.isna(), np.nan,
                                   ((df.male == 1) & (df.BMXWAIST >= 102) | (df.male == 0) & (df.BMXWAIST >= 88)).astype(float))
df["current_daily_smoker"] = ((df.SMQ020 == 1) & (df.SMQ040 == 1)).astype(int)
df["current_smoker_any"] = ((df.SMQ020 == 1) & df.SMQ040.isin([1, 2])).astype(int)
df["former_smoker"] = ((df.SMQ020 == 1) & (df.SMQ040 == 3)).astype(int)
df["edu_more_than_hs"] = np.where(df.DMDEDUC2.isin([1, 2, 3, 4, 5]), (df.DMDEDUC2 >= 4).astype(float), np.nan)
df["race_white"] = (df.RIDRETH1 == 3).astype(int); df["race_black"] = (df.RIDRETH1 == 4).astype(int)
df["race_mexican"] = (df.RIDRETH1 == 1).astype(int); df["race_other"] = df.RIDRETH1.isin([2, 5]).astype(int)
df["marital_never"] = np.where(df.DMDMARTL.isin([1, 2, 3, 4, 5, 6]), (df.DMDMARTL == 5).astype(float), np.nan)
df["marital_widowed"] = np.where(df.DMDMARTL.isin([1, 2, 3, 4, 5, 6]), (df.DMDMARTL == 2).astype(float), np.nan)
df["diabetes_hba1c"] = np.where(df.LBXGH.isna(), np.nan, (df.LBXGH >= 6.5).astype(float))
df["hypertension_measured"] = np.where(df.SBP_mean.isna(), np.nan, ((df.SBP_mean >= 140) | (df.DBP_mean >= 90)).astype(float))
df["log_crp"] = np.log(df.LBXCRP.clip(lower=0.01))
df["log_gtt"] = np.log(df.LBXSGTSI.clip(lower=1))
df["log_creat"] = np.log(df.LBXSCR.clip(lower=0.1))
df["log_trig"] = np.log(df.LBXSTR.clip(lower=10))
df["nlr"] = df.LBXNEPCT / df.LBXLYPCT.replace(0, np.nan)
df["wt4"] = df.WTMEC2YR / 4
drop = ["MORTSTAT", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "SMQ020", "SMQ040", "WTMEC2YR"] + sy + di
df = df.drop(columns=drop)
df.to_csv(OUT, index=False)
print(df.groupby("cycle").agg(n=("SEQN", "size"), events=("event", "sum"), adults40=("RIDAGEYR", lambda x: int((x >= 40).sum()))))
a = df[df.RIDAGEYR >= 40]
print("adults>=40:", len(a), "events", int(a.event.sum()))
print(a.groupby("cycle").agg(n=("SEQN", "size"), events=("event", "sum")))
miss = (a.isna().mean() * 100).round(1).sort_values(ascending=False)
print(miss[miss > 0].to_string())

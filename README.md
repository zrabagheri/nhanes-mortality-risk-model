# NHANES 2001–2008: 10-year all-cause mortality risk model (adults ≥ 40 years)

Code for a pooled-NHANES prediction study: age + sex baseline, elastic-net/lasso Cox model,
a compact 12-variable Cox model, and a gradient-boosting comparator. Models are developed in
NHANES 2001–2006 and temporally externally validated (frozen) in NHANES 2007–2008.

**No participant-level data are included.** Download the public NHANES files and the NCHS
2019 Public-Use Linked Mortality Files yourself (links below).

## Data
- NHANES 2001–2008 (cycles B–E): demographics, blood pressure, body measures, smoking, and
  laboratory files (CBC, biochemistry, CRP, HbA1c, cholesterol, HDL, etc.):
  https://www.cdc.gov/nchs/nhanes/
- NCHS 2019 Public-Use Linked Mortality Files (fixed-width .dat):
  https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
  (SEQN cols 0–5, ELIGSTAT 14, MORTSTAT 15, PERMTH_INT 42–45)

## Run order
```
pip install -r requirements.txt
python build_pool.py <dir_with_xpt_and_mortality_dat> pool_analytic.csv
mkdir -p pool_out figs
python pool_models.py 40 age40        # main analysis (adults >= 40)
python pool_models.py 18 all18        # sensitivity (all adults)
python compact_model2.py              # compact models of 8/12/16/20 variables
python final_compact12.py             # final compact-12 metrics, NRI/IDI, DCA, calibration
python table_stats.py
python figs.py
```
Scripts write to `pool_out/`; the `results/` folder holds the JSON outputs used in the paper
and `figures/` the figures (the scripts write new ones to `figs/`).

`cox_lib.py` is an in-house Cox proportional-hazards implementation (Efron ties) used because
`lifelines` could not be installed in the analysis environment.

## Reproducibility notes
- Preprocessing (winsorisation, median imputation, standardisation) is fitted on the development
  cohort only and applied unchanged to the validation cohort.
- Random seeds are fixed in the scripts.

## Citation
Please cite the accompanying paper (details to be added on publication).

## License
MIT (see LICENSE).

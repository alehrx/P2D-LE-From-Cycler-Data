# Identifying a multi-particle battery model with linearised electrolyte from cycler data alone

Code, data and identified parameter sets of the paper (submitted). The model is a Doyle–Fuller–Newman model solved
with a few particles through the thickness of each electrode and a linearised electrolyte (MPM-LE), written in
dimensionless groups; the groups are identified from the cycler data of commercial NMC532/graphite pouch cells at
−5, 25 and 45 °C and validated on 18 held-out cells. The same pipeline is applied to a second cell type, the LG M50T
21700 NMC811/graphite–SiOx cell, calibrated at 10, 25 and 40 °C and validated on six held-out cells.

## Contents

| Folder | What |
|---|---|
| `paper_notebook.ipynb` | the digital version of the paper: Figs. 4, 8, 9, 10 and Tables 2-9 of Section 3, made by the paper's own scripts from the saved parameter sets (set `RUN_FITS = True` to rerun step 1 and the fit) |
| `srcs/` | the model (`ms_model.py`, `ms_fv.py`: batched multi-slice model with exact finite-volume particles, on a GPU; `electrolyte_modes.py`: the modal linear electrolyte), groups and windows (`identification.py`, `window.py`), the study wrapper (`study.py`), the batched multi-start Levenberg–Marquardt search (`optimizers.py`), the Fisher/collinearity analysis (`identifiability.py`), the dataset readers (`datasets/`: `mohtat.py`, `lgm50.py`), the OCV hysteresis term of the LG cell (`identification.hysteresis_voltage`) |
| `scripts/` | step 0: `fit_lm.py` on `configs/precalibration.json` (pre-calibration) and `screening_local.py` (sensitivities and correlations around it, Table 4); step 1: `step1_window.py` (cathode OCP and electrode windows); step 2: `fit_lm.py` (joint fit); `new_study.py` (a study with your own choice of cells); the paper's figure scripts (`fig_groups.py` Fig. 4, `fig_cases_grid.py` Fig. 8, `error_bins.py` + `fig_cases_errors.py` Fig. 9, `fig_lg_grid.py` Fig. 10); for the LG cell also `hysteresis_c10_gitt.py` (the OCV hysteresis, step 1) and `select_groups_rule.py` (the selection rule of Section 3.3 applied to a screening) |
| `configs/` | the three cases of the paper: `fitA` (measured anode OCP), `fitB` (generic graphite OCP), `caseC` (groups of B, mean capacity), the step-0 studies `precalibration` (the pre-calibration at 25 °C, around which the screening is computed and the fits draw their starts) and `screening` (Table 4); the LG cell: `lgA` (anode OCP of the cell itself), `lgB` (generic blended anode), `lgC` (groups of B, mean capacity), `lg_precalibration` and `lg_screening` (Table 9); `datasets/mohtat.json` and `datasets/lgm50.json` describe the data |
| `results/` | saved results the notebook shows: online capacity (Table 7) and ablation of the calibration cycles (Table 8) |
| `figures/` | Figs. 4, 8, 9 and 10 as made by the notebook |
| `runs/studies/<case>/` | the identified groups (`dynamics.json`), the electrode windows of every cell (`windows.json`), the derived cathode OCP (`ocp_correction.json`) and the step-1 details (`step1.json`); for the LG cases also the OCV hysteresis (`hysteresis.json`) |
| `data/` | the signals of the 21 cells at beginning of life: C/20 cycle (`DB_REAL_Cxx_C20_BOL`), HPPC (`..._HPPC_BOL`) and first ageing cycle (`..._VAL_BOL`), resampled; `cycles.csv` + `config.json` per test; LG M50T: C/10 cycle (`DB_LGM50_<cell>_C10_BOL`), GITT (`..._GITT_BOL`, Expt 4) and first ageing cycles (`..._VAL_BOL`) of the twelve cells used |

Calibration cells: 07, 08, 09 (2C at 25, −5, 45 °C), with the HPPC and C/20 of cell 07; the other 18 cells are held
out.
LG M50T: cells E4A, E4D, E4F (drive-cycle ageing) and E5A, E5D, E5F (1C cycling) at 10, 25 and 40 °C for the windows
and the fit (C/10 and GITT of E4D, first 1C cycles of E5A, E5D, E5F); held out: E4B, E4E, E4G (WLTP part of the first
cycles) and E5B, E5E, E5G (first 1C cycle).

## Reproduce the result figures

```
pip install -r requirements.txt
python make_figures.py                # Fig. 8 and Fig. 9 of the paper, cases A, B, C -> figures/
python make_figures.py --sets B       # one parameter set only
```

## Re-identify

```
python scripts/fit_lm.py configs/precalibration.json --fresh                        # step 0: pre-calibration at 25 C
python scripts/screening_local.py screening precalibration --r 40 --wmin 0.1        # step 0: screening (Table 4)
python scripts/step1_window.py configs/fitB.json                                  # step 1: windows and cathode OCP
python scripts/anchor_window.py fitB fitB                                         # step 1: common window of the calibration cells
python scripts/fit_lm.py configs/fitB.json --fresh                                # step 2: joint fit (~40 min on a GPU)
```

LG M50T (case B):

```
python scripts/fit_lm.py configs/lg_precalibration.json --fresh                     # step 0: pre-calibration at 25 C
python scripts/screening_local.py lg_screening lg_precalibration --r 40 --wmin 0.1  # step 0: screening (Table 9)
python scripts/select_groups_rule.py runs/screening/lg_screening_local/screening_local.json configs/lg_precalibration.json configs/lgB.json lgB
python scripts/step1_window.py configs/lgB.json                                     # step 1: windows and cathode OCP
python scripts/mean_window.py configs/lgB.json                                      # step 1: common window of the window cells
python scripts/hysteresis_c10_gitt.py configs/lgB.json                              # step 1: OCV hysteresis
python scripts/fit_lm.py configs/lgB.json --fresh                                   # step 2: joint fit
```

`scripts/screening.py`, `screening_model.py` and `sensitivity.py` are helpers of `screening_local.py` (the unit cube of
the optimiser, the Morris design, the collinearity index); their own command lines run the earlier global screening,
which the paper does not use.

The configs of the three cases draw their 32 Sobol starts in a box around a pre-calibration,
`"lm": {"center": {"study": "precalibration", "width": 0.4}}`: the groups the pre-calibration
fitted are drawn within +-0.2 of their unit-cube range around its values (diffusion times moved from 25 C to the
fit's reference temperature with each start's own activation energy), the temperature terms over their whole range.

## Solver settings

The current split of each sample is solved by Newton's method with a fixed number of iterations, so that all
candidate parameter sets advance together on the GPU. Set in the `model` block of a config:

```
"newton": [12, 24],          # iterations per sample, and after a change of current
"jump_start": "shift"        # restart from the previous split moved to the new current
```

The defaults (4/8, `"uniform"`, restart from the uniform split) are kept only to reproduce older studies: with
them some candidates do not converge, and parameter sets 0.1 % apart differ by 1-4 mV at 25 C and up to 140 mV at
the end of the cold 2C discharges, which the forward-difference Jacobian of the fit then follows. Check with
`python scripts/check_mode_smoothness.py` (CPU; set `NEWTON`, `NEWTON_JUMP`, `JUMP_START` in the environment).

Other scripts: `scripts/make_case_c.py fitB caseC` (case C: the groups of a fit at the mean capacity of its
calibration cells), `scripts/heldout_summary.py fitA fitB caseC` (held-out first-cycle RMSE per temperature),
`scripts/heldout_capacity_fit.py caseC --mean-study caseC --cap-study fitB --smin 0.96 --smax 1.05` (case C with the
capacity of each held-out cell fitted on its first discharge, Table 7).

## Your own choice of cells

```
python scripts/new_study.py fitB my_study --cal C04 C05 C06 --up C04 C05 C06
python scripts/step1_window.py configs/my_study.json
python scripts/fit_lm.py configs/my_study.json --fresh
python make_figures.py --sets B:fitB X:my_study
```

## Data source

The cell data come from the public dataset of Mohtat et al. (University of Michigan), "UofM pouch cell voltage and
expansion cyclic aging dataset"; cite the dataset and its papers when you use them. The files here are resampled
extracts of the beginning-of-life tests.

The LG M50T data come from the public dataset of Kirkaldy et al., "Lithium-ion battery
degradation: Comprehensive cycle ageing data and analysis for commercial 21700 cells" (J. Power Sources, 2024;
zenodo 10.5281/zenodo.10637534); cite the dataset and its paper when you use them. The files here are resampled
extracts of the beginning-of-life checks and of the first ageing cycles. To prepare them again from the raw files,
set `raw_root` in `configs/datasets/lgm50.json`.

## License

The code is released under the MIT License (`LICENSE`). The data in `data/` are extracts of the public datasets
cited above and remain under the terms of those datasets.

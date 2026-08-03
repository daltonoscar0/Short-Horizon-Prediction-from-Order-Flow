# Phase 0: raw LOBSTER CSVs -> validated event frame. `make test` must run clean from a
# fresh checkout with only the declared deps (no `pip install -e .` step) — src is put on
# PYTHONPATH directly, per DECISIONS.md.
RAW_DIR ?= data/raw/lobster
OUT_DIR ?= data/frames
REPORT ?= reports/data_quality.md
FEATURES_OUT ?= data/features
SEED ?= 0

# Plain `python3` is preferred as-is (a normal single-interpreter checkout needs nothing
# else). On this dev machine specifically, `python3` can resolve ahead of the environment
# that actually has the declared deps depending on shell/login mode (macOS path_helper vs.
# conda's PATH prepend order — see DECISIONS.md), landing on an interpreter without pyarrow.
# Fall back to known-good interpreters only if the first choice can't import the deps; this is
# a no-op everywhere except this specific quirk. `.venv-edge` leads because it is the only
# candidate this repo can create for itself — the anaconda install this project was developed
# against disappeared from the machine mid-session on 2026-08-03, which is exactly the failure
# this probe exists to survive. Every candidate is PROBED, never assumed: a missing
# interpreter falls through instead of failing the target.
PYTHON := $(shell for c in .venv-edge/bin/python3 python3 /opt/anaconda3/bin/python3 python3.12; do \
	command -v $$c >/dev/null 2>&1 && $$c -c "import numpy, pyarrow" >/dev/null 2>&1 && { echo $$c; break; }; \
done)

SELFCHECK_REPORT ?= reports/ofi_selfcheck.md
CORR_REPORT ?= reports/feature_correlations.md
CORR_CSV ?= reports/feature_correlations.csv
LADDER_REPORT ?= reports/ladder.md
FLOOR_REPORT ?= reports/artifact_floor.md
COSTS_REPORT ?= reports/costs.md
TRIALS ?= 2152

export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: data features correlations selfcheck ladder floors costs test

# `data` chains the adapter and the QC report: one command, raw CSVs to a validated event
# frame with a printed/committed quality report (PLAN.md phase-0 milestone). Make aborts the
# whole target if either step exits non-zero, so a hard QC invariant failure fails `make data`.
data:
	$(PYTHON) -m edge.io.lobster --raw $(RAW_DIR) --out $(OUT_DIR) --seed $(SEED)
	$(PYTHON) -m edge.io.quality --raw $(RAW_DIR) --frames $(OUT_DIR) --out $(REPORT) --seed $(SEED)

# Phase 1: the six family modules merged into one table per ticker-day-clock
# (docs/INTERFACE.md sec. 5 / src/edge/features/build.py), then chains straight into the
# correlation table -- PLAN.md's phase-1 milestone is "features.parquet + a correlation
# table", not the parquet alone, so `make features` produces both in one command. Same
# abort-the-whole-target behaviour as `data`: either step exiting non-zero fails the target.
features:
	$(PYTHON) -m edge.features.build --frames $(OUT_DIR) --out $(FEATURES_OUT) --seed $(SEED)
	$(PYTHON) -m edge.analysis.correlations --features $(FEATURES_OUT) --out-md $(CORR_REPORT) --out-csv $(CORR_CSV) --seed $(SEED)

# Standalone re-run of the correlation table alone (e.g. after a `data/features` rebuild
# done elsewhere) -- same command `features` chains into above.
correlations:
	$(PYTHON) -m edge.analysis.correlations --features $(FEATURES_OUT) --out-md $(CORR_REPORT) --out-csv $(CORR_CSV) --seed $(SEED)

# Phase-1 gate (PLAN.md L2 / FEATURES.md sec. 1): reproduce Cont-Kukanov-Stoikov's
# Delta-mid ~ OFI relation on the built features and fail loudly if it doesn't hold --
# "if you can't reproduce a 20-year-old linear result, stop and fix the features." Exits
# non-zero (and Make aborts) on any of the three CKS checks failing, same as `data`.
selfcheck:
	$(PYTHON) -m edge.analysis.ofi_selfcheck --features $(FEATURES_OUT) --raw $(RAW_DIR) --out $(SELFCHECK_REPORT) --seed $(SEED)

# Phase 2, the contribution: the L0-L4 ladder under protocols A/B/C/D, the embargo
# sensitivity sweep, the superseded in-sample-calibration diagnostic kept beside the corrected
# columns so the size of the 2026-08-02 correction is visible rather than asserted, and the
# calibration-rule control. `--holdout-control` is NOT optional here: the headline `A' - C`
# numbers come from the table it emits, and for one revision this target regenerated the report
# without it -- a published command that did not reproduce the published report (CLAUDE.md
# rule 7, caught by the fourth red team).
ladder:
	$(PYTHON) -m edge.harness.run --features $(FEATURES_OUT) --out $(LADDER_REPORT) \
		--seed $(SEED) --embargo-sweep --in-sample-scale --holdout-control

# The artifact floor the ladder's gaps are read against: one predictor, fitted once, scored
# unchanged under every protocol, at three rung magnitudes (a floor does not transfer between
# them). Independent of `ladder` -- run either alone.
floors:
	$(PYTHON) -m edge.harness.floors --features $(FEATURES_OUT) --out $(FLOOR_REPORT) \
		--seed $(SEED)

# Phase 3: latency/fill model, break-even cost table, DM tests, deflated Sharpe. TRIALS is
# the honest trial count from RESULTS.md's log -- passing a smaller one silently undeflates
# the Sharpe, so it is a variable here rather than a default buried in the module.
costs:
	$(PYTHON) -m edge.costs.run --features $(FEATURES_OUT) --out $(COSTS_REPORT) \
		--seed $(SEED) --trials $(TRIALS) --tickers INTC,AMZN --clocks calendar,event \
		--horizons 1s,5s

test:
	$(PYTHON) -m unittest discover -s tests -v

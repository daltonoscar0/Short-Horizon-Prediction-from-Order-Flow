# Phase 0: raw LOBSTER CSVs -> validated event frame. `make test` must run clean from a
# fresh checkout with only the declared deps (no `pip install -e .` step) — src is put on
# PYTHONPATH directly, per DECISIONS.md.
RAW_DIR ?= data/raw/lobster
OUT_DIR ?= data/frames
REPORT ?= reports/data_quality.md
SEED ?= 0

# Plain `python3` is preferred as-is (a normal single-interpreter checkout needs nothing
# else). On this dev machine specifically, `python3` can resolve ahead of the environment
# that actually has the declared deps depending on shell/login mode (macOS path_helper vs.
# conda's PATH prepend order — see DECISIONS.md), landing on a <3.10 interpreter without
# pyarrow. Fall back to known-good interpreters only if the first choice can't import the
# deps; this is a no-op everywhere except this specific quirk.
PYTHON := $(shell for c in python3 /opt/anaconda3/bin/python3 python3.12; do \
	command -v $$c >/dev/null 2>&1 && $$c -c "import numpy, pyarrow" >/dev/null 2>&1 && { echo $$c; break; }; \
done)

export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: data test

data:
	$(PYTHON) -m edge.io.lobster --raw $(RAW_DIR) --out $(OUT_DIR) --seed $(SEED)
	$(PYTHON) -m edge.io.quality --frames $(OUT_DIR) --out $(REPORT) --seed $(SEED)

test:
	$(PYTHON) -m unittest discover -s tests -v

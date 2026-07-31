# Phase 0: raw LOBSTER CSVs -> validated event frame. `make test` must run clean from a
# fresh checkout with only the declared deps (no `pip install -e .` step) — src is put on
# PYTHONPATH directly, per DECISIONS.md.
RAW_DIR ?= data/raw/lobster
OUT_DIR ?= data/frames
SEED ?= 0

export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: data test

data:
	python3 -m edge.io.lobster --raw $(RAW_DIR) --out $(OUT_DIR) --seed $(SEED)

test:
	python3 -m unittest discover -s tests -v

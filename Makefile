.PHONY: setup setup-gpu lint test verify data scale gpu bench audit ci

PY ?= python
TORCH_CPU = https://download.pytorch.org/whl/cpu
TORCH_CU130 = https://download.pytorch.org/whl/cu130
SCALE_ARGS = --model scaled --data tinystories --steps 15 --global-batch 32 --seq-len 128 --threads-per-rank 2 --warmup-steps 3
GPU_ARGS = --device cuda --model scaled --data tinystories --steps 300 --global-batch 32 --seq-len 128 --lr 1e-3 --warmup-steps 20

# CPU build of PyTorch (enough for every test and for the gloo multi-process runs). setuptools
# comes from PyPI first: the PyTorch index only carries an old copy (78.1.0) with known advisories.
setup:
	$(PY) -m pip install --upgrade pip setuptools
	$(PY) -m pip install torch --index-url $(TORCH_CPU)
	$(PY) -m pip install -e ".[dev]"

# CUDA 13.0 build of PyTorch, for the single-GPU run.
setup-gpu:
	$(PY) -m pip install --upgrade pip setuptools
	$(PY) -m pip install torch --index-url $(TORCH_CU130)
	$(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# 57 tests; the distributed ones start 2 to 4 gloo processes on the CPU.
test:
	$(PY) -m pytest -q

# Ring all-reduce against the expected sum and gloo's all-reduce, 4 ranks.
verify:
	dtlab verify --world-size 4

# Pinned TinyStories validation file (19 MB, SHA-256 checked) into .tmp/data.
data:
	dtlab data

# Strong scaling on the CPU: the same global batch on 1, 2 and 4 gloo ranks.
scale: data
	dtlab scale --world-sizes 1,2,4 $(SCALE_ARGS) --name cpu-scale

# One GPU run, then two ranks sharing that GPU (needs make setup-gpu).
gpu: data
	dtlab train --world-size 1 $(GPU_ARGS) --name gpu-w1
	dtlab train --world-size 2 $(GPU_ARGS) --name gpu-w2

bench: scale

# Known vulnerabilities; local version tags (+cpu, +cu130) are stripped so PyPI data applies.
audit:
	$(PY) -m pip freeze --exclude-editable | sed 's/+[a-z0-9]*$$//' > .tmp/requirements-audit.txt
	$(PY) -m pip_audit -r .tmp/requirements-audit.txt --no-deps --disable-pip --cache-dir .tmp/pip-audit

ci: setup lint test

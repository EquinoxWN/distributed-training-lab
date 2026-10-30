# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+ and GNU make. `make setup` installs the CPU build of PyTorch, which is enough
for every test; `make setup-gpu` installs the CUDA 13.0 build. Caches, data and run results stay
in git-ignored folders inside the repo (`.venv/`, `.tmp/`).

```bash
make setup   # install dependencies
make lint    # ruff, ruff format, mypy --strict
make test    # full test suite (starts 2 to 4 gloo processes)
make verify  # ring all-reduce check on 4 ranks
make audit   # known-vulnerability check (as in CI)
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change; a new strategy must match single-process
  training the way `tests/test_distributed.py` checks DDP.
- Update `docs/results/` when numbers change, with the machine and command used.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).

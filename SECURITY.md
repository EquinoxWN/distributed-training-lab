# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/distributed-training-lab/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** a tampered or oversized dataset download being used, downloads that are not over
  HTTPS, run output written outside `.tmp/runs`, and configuration values that are not validated.
- **Out of scope:** the gloo process group is meant for one trusted machine; exposing its port to
  an untrusted network is not supported. Training quality and speed are measurements, not
  vulnerabilities.

## How this repository protects itself

- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- CI runs lint, type checks, tests and a known-vulnerability check (pip-audit) on every push and pull request.

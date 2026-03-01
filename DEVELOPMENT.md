# Development notes

## Scoped strict warnings

This repository uses a gradual "warnings-as-errors" approach.

- Local (scoped example):
  tox -e py312 -- tests/test_backend.py -W error::DeprecationWarning -W error::FutureWarning

- CI (scoped example):
  The job `strict-warnings (alias_utils)` runs the same command on a focused test subset.

To extend the pattern to another area, add a new focused test target (or marker) and introduce a new CI job that runs only that subset with `-W error::DeprecationWarning` and `-W error::FutureWarning`.

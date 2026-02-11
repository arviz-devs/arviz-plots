# Contributing guidelines

## Before contributing

Welcome to arviz-plots! Before contributing to the project,
make sure that you **read our code of conduct** (`CODE_OF_CONDUCT.md`).

## Contributing code

1. Set up a Python development environment
   (advice: use [venv](https://docs.python.org/3/library/venv.html),
   [virtualenv](https://virtualenv.pypa.io/), or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
2. Install tox: `python -m pip install tox tox-gh-actions`
3. Clone the repository
4. Start a new branch off main: `git switch -c new-branch main`
5. Make your code changes
6. Check that your code follows the style guidelines of the project: `tox -e check`
7. (optional) Build the documentation: `tox -e docs`
8. (optional) Run the tests: `tox -e py312`
   (use py311/py312/py313 depending on the Python you are using)
9. Commit, push, and open a pull request!

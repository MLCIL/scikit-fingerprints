# Contributing to scikit-fingerprints library

Thank you for your interest in contributing to scikit-fingerprints library! We welcome contributions from
the community to make our project even better.

## Getting Started

Before you start contributing, please take a moment to review the following guidelines to ensure a smooth and productive
collaboration.

### Reporting Issues

If you find a bug, have a question, or want to propose a new feature, please open an issue on
our [GitHub repository](https://github.com/scikit-fingerprints/scikit-fingerprints.git). Be sure to include as much detail as
possible, such as a clear description, steps to reproduce the issue, and any relevant screenshots or error messages.

### Pull Requests

We encourage you to submit pull requests (PRs) to improve our project. To do so, follow these steps:

1. Set up your development environment by following the [instructions](#development-setup-linux). In particular,
   make sure that pre-commit hooks are working.

2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b my-feature

3. Make your changes. Make sure to follow the existing coding style and conventions.
   Commit changes with clear and concise commit messages.

4. Push your changes to your forked repository:
    ```bash
    git push origin my-feature

5. Open a pull request on our GitHub repository, make sure to fill the PR template.
   PR name should be well-formatted, concise, and clearly define what you did, e.g.
   "Implemented X", "Fixed Y", "Added Z and Q".

6. Participate in the review process by addressing any feedback or comments on your PR.

### Development Setup (Linux)

Prerequisites:

- Git
- Python 3.10 or higher
- uv
- Pandoc

To set up the development environment, follow these steps:

1. Install prerequisites from above.
2. Fork the repository on GitHub.
3. Clone your forked repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
4. Run make to set up the development environment:
   ```bash
   make setup
   ```
   This command will set up a virtual environment and install all the necessary dependencies.
5. Activate the virtual environment. On Linux, it is:
   ```bash
   source .venv/bin/activate
   ```
6. That's it! You're ready to start developing.

If you are using PyCharm, then mark `tests` directory as tests sources root. Also make sure that
PyTest is configured as the default test runner. This will make running them easier from UI.

---
**NOTE**

For other operating systems, please refer to Makefile for the commands to run.

---

### Testing

Before submitting a pull request, make sure to run the tests to ensure that your changes do not introduce
regressions.
To run tests, execute `make test` in the main directory of the repository.
If necessary, add new tests to cover your code. Also, please be sure that you do not violate any code style
requirements (you can check it by running pre-commit on staged files).

### Documentation

If you are contributing new features or changes, you also need to update the documentation to reflect your changes.
Most of the changes will be covered by small changes in docstrings. However adding new functionality may require to
change overall docs structure. You can find it in the `docs` directory.

Run `make docs` in the main directory of the repository to build the documentation. This command will generate
HTML files in the `docs/_build/html` directory.

To view the documentation, open the `docs/_build/html/index.html` file in your browser.

We care about the quality of our documentation. Before merging your Pull Request, there must be no warning regarding docs.
You can check for any with `make doctest` to check whole documentation or just `uv run python -m doctest <PATH_TO_MODULE>` 
for specific file.

### Releasing
To release scikit-fingerprints open a GitHub release with tag named 'vA.B.C'
where each letter stands for version number. Fill the release notes and submit the release.
Then, the version will be automatically sourced from tag by GH action and released to PyPI.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for your contribution! Your efforts help make the scikit-fingerprints library better for everyone.

## LLMs and AI agents policy

Primarily or purely LLM-generated contributions are **strictly forbidden**. AI agents and in particular 
autonomous PRs are not allowed. PRs submitted this way will result in a ban and report to GitHub as spam.

Usage of LLMs is allowed only for polishing code, detecting bugs, and other supportive tasks. All PRs
must be human-made and human-supervised. PR author takes full responsibility for the code.

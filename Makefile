.PHONY: setup docs doctest run-doctest test test-coverage help
.DEFAULT_GOAL := help

setup: ## Install development dependencies
	@# check if uv is installed
	@uv --version >/dev/null 2>&1 || (echo "uv is not installed, please install it" && exit 1)

	@# check if pandoc (for docs) is installed
	@pandoc --version >/dev/null 2>&1 || (echo "Pandoc is not installed. Please install it from https://pandoc.org/" && exit 1)

	@# install dependencies
	uv sync --group dev --group docs --extra neural
	uv run pre-commit install

docs: ## Re-generate documentation
	uv run $(MAKE) -C docs html

# detect if datasets changed for tests
# on CI, PR_BASE_SHA / PR_HEAD_SHA pin the exact PR base and head, which is
# robust against the auto-generated merge commit and a moving origin/master
# locally compare current branch against origin/master
define DATASETS_CHANGED
{ \
	if [ -n "$$PR_BASE_SHA" ] && [ -n "$$PR_HEAD_SHA" ]; then \
		git diff --name-only "$$PR_BASE_SHA...$$PR_HEAD_SHA" ;\
	else \
		git diff --name-only origin/master...HEAD ;\
	fi ;\
	git diff --name-only --cached ;\
	git diff --name-only ;\
} | grep -qE '^(skfp|tests)/datasets/'
endef

test: ## Run tests
	uv run ruff check

	@# datasets tests are slow, so we run them only if Git indicates change there
	@if $(DATASETS_CHANGED); then \
	  echo "Datasets changed, running all tests" ;\
	  uv run pytest tests ;\
	else \
	  echo "Skipping datasets tests" ;\
	  uv run pytest tests --ignore=tests/datasets ;\
	fi

test_with_datasets: ## Run tests, always including dataset tests
	uv run ruff check
	zizmor .github
	uv run pytest tests

test-coverage: ## Run tests and calculate test coverage
	-mkdir .tmp_coverage_files
	uv run pytest --cov=skfp tests
	-rm -rf .tmp_coverage_files

doctest: docs run-doctest ## Build docs and run documentation tests

run-doctest: ## Run documentation tests
	@if $(DATASETS_CHANGED); then \
	  echo "Datasets changed, running all doctests" ;\
	  uv run $(MAKE) -C docs doctest ;\
	else \
	  echo "Skipping datasets doctests" ;\
	  SKFP_SKIP_DATASET_DOCTESTS=1 uv run $(MAKE) -C docs doctest ;\
	fi

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

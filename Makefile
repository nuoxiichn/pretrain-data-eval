.PHONY: check test lint contracts

check: lint contracts test

test:
	python -m pytest -q

lint:
	ruff check pretrain_data_eval stages tests
	python -m compileall -q pretrain_data_eval stages scripts tests
	bash -n scripts/*.sh scripts/archive/*.sh
	git diff --check

contracts:
	python -m pytest -q tests/contracts

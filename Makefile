.PHONY: install run smoke validate test

install:
	python -m pip install -r requirements.txt

run:
	python src/run_pipeline.py --project-dir .

smoke:
	python src/run_pipeline.py --project-dir . --n-trials 8 --cv-sample-frac 0.20 --sarimax-store-count 3

validate:
	python src/validate_artifacts.py --project-dir .

test:
	python -m pytest

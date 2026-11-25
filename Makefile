.PHONY: install test notebook reproduce clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

notebook:
	jupytext --to ipynb notebooks/01_map_matching.py
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_map_matching.ipynb

reproduce: test notebook

clean:
	rm -rf .pytest_cache notebooks/.ipynb_checkpoints

.PHONY: install test bench bench-offline live clean
install:
	pip install -e . 
test:
	PYTHONPATH=src pytest -q tests/
bench:
	PYTHONPATH=src python experiments/exp01_stochastic.py
	PYTHONPATH=src python experiments/exp02_adversarial.py
	PYTHONPATH=src python experiments/exp03_linear.py
	PYTHONPATH=src python experiments/exp05_hidden_patterns.py
bench-offline:
	PYTHONPATH=src python experiments/exp04_live_market.py --synthetic --folds 4
live:
	PYTHONPATH=src python experiments/exp04_live_market.py --folds 4 --cost_bps 5.0
clean:
	rm -rf results/*.json results/*.csv __pycache__ .pytest_cache

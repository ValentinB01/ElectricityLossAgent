# Every step of the producers module, end to end (CLAUDE.md section 13). Run from the repository root.
PYTHON ?= python

.PHONY: help setup data data-test train eval eval-final plan test clean

help:
	@echo "setup       install the pinned requirements"
	@echo "data        Kelmarsh 2019 + Kaggle solar + Open-Meteo history into data/"
	@echo "data-test   Kelmarsh 2020, the test year"
	@echo "train       train every model into models/"
	@echo "eval        score on validation and run the dev backtest, into results/"
	@echo "eval-final  reproduce the test-year results in the README (settings are frozen)"
	@echo "plan        today's calendar for Fantanele and Calugareni (live Open-Meteo forecast)"
	@echo "test        optimizer tests"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.prepare --year 2019
	$(PYTHON) -m src.data.prepare --pv

data-test:
	$(PYTHON) -m src.data.prepare --year 2020

train:
	$(PYTHON) -m src.producers.expected_power --train
	$(PYTHON) -m src.producers.underperformance --train
	$(PYTHON) -m src.producers.downtime_cost --train
	$(PYTHON) -m src.producers.solar --train

eval:
	$(PYTHON) -m src.producers.expected_power --eval
	$(PYTHON) -m src.producers.underperformance --eval
	$(PYTHON) -m src.producers.downtime_cost --eval
	$(PYTHON) -m src.producers.solar --eval
	$(PYTHON) -m src.decision.backtest --dev

eval-final: data-test
	$(PYTHON) -m src.producers.expected_power --eval --final
	$(PYTHON) -m src.producers.underperformance --eval --final
	$(PYTHON) -m src.producers.downtime_cost --eval --final
	$(PYTHON) -m src.producers.solar --eval --final
	$(PYTHON) -m src.decision.backtest --final

plan:
	$(PYTHON) -m src.decision.optimizer --site fantanele calugareni

test:
	$(PYTHON) -m pytest

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

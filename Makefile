.PHONY: help setup data data-full train eval eval-agent agent demo test

help:
	@echo "Available commands:"
	@echo "  make setup        Install project dependencies"
	@echo "  make agent        Start local FastAPI agent service on 127.0.0.1:8000"
	@echo "  make demo         Launch Streamlit interface"
	@echo "  make eval-agent   Run end-to-end agent scenario evaluation suite"
	@echo "  make test         Run test suite across all components"

setup:
	python -m pip install -r requirements.txt

agent:
	python -m uvicorn src.agent.api:app --host 127.0.0.1 --port 8000

demo:
	python -m streamlit run src/app/dashboard.py

eval-agent:
	python -m pytest tests/test_agent_scenarios.py

test:
	python -m pytest tests/


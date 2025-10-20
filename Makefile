.PHONY: setup train api docker-run fmt test

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.cli.train_and_eval

api:
	uvicorn app.main:app --reload --port 8000

docker-build:
	docker build -t local/diabetes-ml:dev .

docker-run:
	docker run --rm -p 8000:8000 local/diabetes-ml:dev

fmt:
	ruff check . --fix

test:
	pytest -q

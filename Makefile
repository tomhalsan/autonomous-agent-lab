.PHONY: up down logs ps test lint clean
up:
	docker compose up --build -d
down:
	docker compose down
logs:
	docker compose logs -f
ps:
	docker compose ps
test:
	python -m pytest
lint:
	python -m ruff check .
clean:
	docker compose down -v --remove-orphans

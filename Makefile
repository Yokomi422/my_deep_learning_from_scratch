fmt:
	poetry run black .
	poetry run isort .
lint:
	poetry run mypy .
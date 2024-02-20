fmt:
	poetry run black .
	poetry run isort .
lint:
	poetry run flake8p .
	poetry run mypy .
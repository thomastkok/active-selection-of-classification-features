init:
		pip install -r requirements.txt

start:
		python -m src

format:
		black .

test:
		pytest
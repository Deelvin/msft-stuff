# Directories containing Python code to format and lint.
PYTHON_DIRECTORIES := utils

format: # Formats the Python code.
	isort ${PYTHON_DIRECTORIES}
	black ${PYTHON_DIRECTORIES}

lint: # Checks the Python code for correct formatting and style.
	isort  --check-only              ${PYTHON_DIRECTORIES}
	black  --check                   ${PYTHON_DIRECTORIES}
	flake8 --config=.flake8          ${PYTHON_DIRECTORIES}
	mypy   --config-file=.mypy.ini   ${PYTHON_DIRECTORIES}

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ece324-TANGO
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pixi install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	pixi run flake8 ece324_tango
	pixi run isort --check --diff ece324_tango
	pixi run black --check ece324_tango

## Format source code with black
.PHONY: format
format:
	pixi run isort ece324_tango
	pixi run black ece324_tango



## Run tests
.PHONY: test
test:
	pixi run pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	@echo ">>> Pixi environment will be created when running 'make requirements'"
	
	@echo ">>> Activate with:\npixi shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	pixi run python -m ece324_tango.dataset

## Train ASCE MAPPO prototype
.PHONY: train_asce
train_asce: requirements
	pixi run train-asce

## Evaluate ASCE MAPPO prototype against baselines
.PHONY: eval_asce
eval_asce: requirements
	pixi run eval-asce


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

# update with python3 or pip3, for example, if needed
PYTHON = python
PIP = pip
RM = rm


# run the main module (effectively runs __main__.py)
run:
	$(PYTHON) -m fire_eye


# install the required third part packages
init:
	$(PIP) install -r requirements.txt


# -b option suppress print outputs
test:
	$(PYTHON) -m unittest discover -b


# no -b option, so printf should work
test-verbose:
	$(PYTHON) -m unittest discover


# finish up/fix... This only removes from top level
# remove __pycache__ stuff,
# empty scraped data dir
# empty preprocessed data dir
clean:
	$(RM) -rf *__pycache__


.PHONY: init run test clean

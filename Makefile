# update with python3 or pip3, for example, if needed
PYTHON = python
PIP = pip
RM = rm


# install the required third part packages
init:
	$(PIP) install -r requirements.txt


# run the main module (effectively runs __main__.py)
run:
	$(PYTHON) -m fire_eye


# -b option suppress print outputs
test:
	$(PYTHON) -m unittest discover -b


# finish up/fix... This only removes from top level
clean:
	$(RM) -rf *__pycache__


.PHONY: init run test clean

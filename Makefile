PYTHON ?= python3
.PHONY: test train report evaluate draw
test:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $(PYTHON) -m unittest discover -s tests -v
train:
	$(PYTHON) train.py
report:
	$(PYTHON) report.py
evaluate:
	$(PYTHON) predict.py

draw:
	$(PYTHON) app.py

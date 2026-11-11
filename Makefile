.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
.PHONY: ajuda ambiente validar testes artigo reproduzir
ajuda:
	@echo "ambiente | validar | testes | artigo (reproduzir)"
	@echo "artigo: refazer o estudo completo a partir dos dois CSVs brutos"

ambiente:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

validar:
	$(VENV) -m pip check
	$(VENV) -B src/validar_ambiente.py

testes:
	$(VENV) -B -m unittest discover -s tests -v

PROCESSOS ?= 1
artigo:
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(VENV) -B -m experimentos.artigo.executar --raw "$(BRUTOS)" --destino "$(DESTINO)" --processos "$(PROCESSOS)"

reproduzir: artigo

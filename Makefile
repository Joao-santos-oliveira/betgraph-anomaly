.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
ESTUDO := main
RUN = $(VENV) -B scripts/pipeline.py $(ESTUDO) --raw "$(BRUTOS)" --destino "$(DESTINO)"

.PHONY: ajuda ambiente validar testes reproduzir historico
ajuda:
	@echo "ambiente    instalar as dependências CPU em Python 3.12"
	@echo "validar     verificar a instalação"
	@echo "testes      executar os testes desta branch"
	@echo "reproduzir  reconstruir os resultados da branch desde os dois CSVs brutos"
	@echo "historico    reproduzir também os pilotos e a validação ampliada"
ambiente:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -r requirements.txt

validar:
	$(VENV) -m pip check
	$(VENV) -B src/validar_ambiente.py

testes:
	$(VENV) -B -m unittest discover -s tests -v

reproduzir:
	$(RUN)

historico:
	$(RUN) --historico

.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
ESCOPO ?= completo
CENARIO ?= original
INJECAO ?= 40101
EPOCAS ?= 100
DETECTOR ?= principal
GRUPO ?= todos
ESTUDO := v2
RUN = $(VENV) -B scripts/pipeline.py $(ESTUDO) --raw "$(BRUTOS)" --destino "$(DESTINO)"

.PHONY: hibrido embeddings-if comparar redes baselines anomalias-estrutura anomalias-volume ajuda ambiente validar testes reproduzir preparar
ajuda:
	@echo "hibrido | embeddings-if: comparar o detector principal e IF nos embeddings"
	@echo "comparar | redes | baselines: ESCOPO=completo ou 5000"
	@echo "anomalias-volume | anomalias-estrutura: 50 alvos, GRUPO=todos, redes ou baselines"
	@echo "ambiente    instalar as dependências CPU em Python 3.12"
	@echo "validar     verificar a instalação"
	@echo "testes      executar os testes desta branch"
	@echo "reproduzir  reconstruir os resultados da branch desde os dois CSVs brutos"
	@echo "preparar    conferir brutos, coorte e intervenções antes dos treinos"
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

preparar:
	$(RUN) --preparar-apenas
OPCOES = --detector "$(DETECTOR)" --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)"

comparar:
	$(RUN) --modelos todos $(OPCOES) --cenario "$(CENARIO)"

baselines:
	$(RUN) --modelos baselines $(OPCOES) --cenario "$(CENARIO)"

redes:
	$(RUN) --modelos redes $(OPCOES) --cenario "$(CENARIO)"

anomalias-volume:
	$(RUN) --modelos "$(GRUPO)" $(OPCOES) --cenario volume

anomalias-estrutura:
	$(RUN) --modelos "$(GRUPO)" $(OPCOES) --cenario estrutura

hibrido:
	$(RUN) --modelos redes --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)" --cenario "$(CENARIO)" --detector principal

embeddings-if:
	$(RUN) --modelos redes --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)" --cenario "$(CENARIO)" --detector if

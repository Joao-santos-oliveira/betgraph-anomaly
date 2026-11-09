.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
ESCOPO ?= completo
CENARIO ?= original
INJECAO ?= 40101
EPOCAS ?= 60
DETECTOR ?= principal
GRUPO ?= todos
SEED_S1_S2 ?= 30101
ESTUDO := v3
RUN = $(VENV) -B scripts/pipeline.py $(ESTUDO) --raw "$(BRUTOS)" --destino "$(DESTINO)"

.PHONY: pares-arestas embeddings-if anomalias-combinadas decoys comparar redes baselines anomalias-estrutura anomalias-volume ajuda ambiente validar testes reproduzir preparar
ajuda:
	@echo "pares-arestas | embeddings-if: comparar o detector principal e IF nos embeddings"
	@echo "anomalias-combinadas | decoys: S1 e S2 em ESCOPO=completo ou 5000"
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
	$(VENV) -B scripts/preparar_referencia.py
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

anomalias-combinadas:
	$(RUN) --modelos "$(GRUPO)" --escopo "$(ESCOPO)" --epocas "$(EPOCAS)" --detector "$(DETECTOR)" --injecao "$(SEED_S1_S2)" --cenario combinado

decoys:
	$(RUN) --modelos "$(GRUPO)" --escopo "$(ESCOPO)" --epocas "$(EPOCAS)" --detector "$(DETECTOR)" --injecao "$(SEED_S1_S2)" --cenario decoys

pares-arestas:
	$(RUN) --modelos redes --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)" --cenario "$(CENARIO)" --detector principal

embeddings-if:
	$(RUN) --modelos redes --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)" --cenario "$(CENARIO)" --detector if

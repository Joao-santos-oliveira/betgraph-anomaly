.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
ESCOPO ?= completo
CENARIO ?= original
INJECAO ?= 40101
EPOCAS ?= 100
GRUPO ?= todos
ESTUDO := main
RUN = $(VENV) -B scripts/pipeline.py $(ESTUDO) --raw "$(BRUTOS)" --destino "$(DESTINO)"

.PHONY: comparar baselines redes anomalias-volume anomalias-estrutura ajuda ambiente validar testes reproduzir historico baselines-5000 redes-5000 comparar-5000 escala-completa
ajuda:
	@echo "artigo: reproduzir o estudo da GNN V2 com a ablação a partir dos dois CSVs brutos"
	@echo "comparar | baselines | redes: dia completo por padrão (ESCOPO=5000 para a amostra)"
	@echo "anomalias-volume | anomalias-estrutura: 50 alvos; GRUPO=todos, baselines ou redes"
	@echo "ambiente    instalar as dependências CPU em Python 3.12"
	@echo "validar     verificar a instalação"
	@echo "testes      executar os testes desta branch"
	@echo "baselines-5000  relacional e IsolationForest na amostra"
	@echo "redes-5000      GNN e MLP na mesma amostra"
	@echo "comparar-5000   quatro métodos na mesma execução"
	@echo "escala-completa benchmark do dia completo (alias de reproduzir)"
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

baselines-5000:
	$(RUN) --modelos baselines --escopo 5000 --cenario "$(CENARIO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)"

redes-5000:
	$(RUN) --modelos redes --escopo 5000 --cenario "$(CENARIO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)"

comparar-5000:
	$(RUN) --modelos todos --escopo 5000 --cenario "$(CENARIO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)"

escala-completa: reproduzir

OPCOES = --escopo "$(ESCOPO)" --injecao "$(INJECAO)" --epocas "$(EPOCAS)"

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

PROCESSOS ?= 1
.PHONY: artigo
artigo:
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(VENV) -B -m experimentos.artigo.executar --raw "$(BRUTOS)" --destino "$(DESTINO)" --processos "$(PROCESSOS)"

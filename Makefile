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
# Evidências do artigo (scripts/evidencias)
REPETICOES ?= 3
PYTHON_FIGURAS ?= $(VENV)
FIGURAS_OPCOES ?=
EXECUCAO_V3 ?= $(lastword $(sort $(wildcard data/benchmark/execucoes/v3-*)))

.PHONY: comparar baselines redes anomalias-volume anomalias-estrutura ajuda ambiente validar testes reproduzir historico baselines-5000 redes-5000 comparar-5000 escala-completa figuras-ambiente benchmark evidencias-dados figuras evidencias
ajuda:
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
	@echo "benchmark       método principal e comparadores por etapa, na mesma máquina (REPETICOES=3)"
	@echo "evidencias      tabela comparativa, seleção dos subgrafos e figuras em PDF/PNG"
	@echo "figuras-ambiente instalar plotly e kaleido (a exportação precisa de um navegador Chromium)"
# Sem o pacote python3.12-venv o ensurepip falha; nesse caso a venv é criada sem pip e o pip vem do bootstrap oficial.
ambiente:
	$(PYTHON) -m venv .venv || { rm -rf .venv; $(PYTHON) -m venv --without-pip .venv && curl -sSfL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -; }
	$(VENV) -m pip install -r requirements.txt

figuras-ambiente:
	$(PYTHON_FIGURAS) -m pip install -r requirements-figuras.txt

benchmark:
	$(VENV) -B scripts/evidencias/benchmark.py --repeticoes "$(REPETICOES)" --raw "$(BRUTOS)" --manter-execucoes

evidencias-dados:
	$(VENV) -B scripts/evidencias/tabela.py --benchmark benchmark/resumo.csv
	$(VENV) -B scripts/evidencias/subgrafos.py selecionar --execucao "$(EXECUCAO_V3)"

figuras:
	$(PYTHON_FIGURAS) -B scripts/evidencias/graficos.py $(FIGURAS_OPCOES)
	$(PYTHON_FIGURAS) -B scripts/evidencias/subgrafos.py desenhar $(FIGURAS_OPCOES)

evidencias: evidencias-dados figuras

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

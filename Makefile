.DEFAULT_GOAL := ajuda
PYTHON ?= python3.12
VENV ?= .venv/bin/python
BRUTOS ?= data/raw
DESTINO ?= data/execucoes
PYTHON_FIGURAS ?= .venv-figuras/bin/python
TABELAS_OPCOES ?=
.PHONY: ajuda ambiente validar testes artigo reproduzir pareado artigo-original dados-artigo figuras-ambiente tabelas graficos graficos-publicacao conteudo benchmark
ajuda:
	@echo "ambiente | validar | testes | artigo (reproduzir) | pareado"
	@echo "artigo: refazer o estudo completo a partir dos dois CSVs brutos"
	@echo "artigo-original: treinar só o dia real (3 seeds)"
	@echo "dados-artigo RODADA=<pasta>: extrair da execução os dados das tabelas e gráficos"
	@echo "figuras-ambiente | tabelas | graficos | graficos-publicacao | conteudo (os três)"
	@echo "benchmark REPETICOES=3: tempo e memória por etapa (depende da máquina)"

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

# Só o dia real (3 treinos em vez de 12): cobertura total, scores das ligações reservadas e embeddings.
artigo-original:
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(VENV) -B -m experimentos.artigo.executar --raw "$(BRUTOS)" --destino "$(DESTINO)" --processos "$(PROCESSOS)" --apenas-original


# Conteúdo do artigo -----------------------------------------------------------------------------
# Tabelas e gráficos leem só resultados/artigo: não dependem da máquina nem treinam nada.

dados-artigo:
	@test -n "$(RODADA)" || (echo "Informe RODADA com a pasta gerada por make artigo ou make artigo-original"; exit 1)
	$(VENV) -B -m experimentos.artigo.extrair "$(RODADA)"

figuras-ambiente:
	$(PYTHON) -m venv .venv-figuras
	$(PYTHON_FIGURAS) -m pip install -r requirements-figuras.txt
	@echo "Para exportar PNG e PDF, o kaleido precisa de um Chrome: $(PYTHON_FIGURAS) -c 'import kaleido; kaleido.get_chrome_sync()'"

# TABELAS_OPCOES=--sem-imagens gera só CSV e LaTeX (sem navegador)
tabelas:
	$(PYTHON_FIGURAS) -B -m experimentos.conteudo.tabelas $(TABELAS_OPCOES)

# GRAFICOS_OPCOES=--sem-imagens gera só CSV, HTML e LaTeX; --apenas G1 G3 limita a lista
graficos:
	$(PYTHON_FIGURAS) -B -m experimentos.conteudo.graficos $(GRAFICOS_OPCOES)

# Versão de publicação dos gráficos (tamanho de impressão, PNG 600 dpi, PDF, SVG, HTML, validação e comparação)
graficos-publicacao:
	$(PYTHON_FIGURAS) -B -m experimentos.conteudo.graficos_publicacao $(GRAFICOS_OPCOES)

conteudo: tabelas graficos graficos-publicacao

# Único alvo que depende da máquina: tempo e pico de memória por etapa (Linux ou WSL).
REPETICOES ?= 3
benchmark:
	$(VENV) -B -m experimentos.conteudo.benchmark --raw "$(BRUTOS)" --repeticoes "$(REPETICOES)"

pareado:
	@test -n "$(RODADA)" || (echo "Informe RODADA com a pasta gerada por make artigo"; exit 1)
	OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(VENV) -B -m experimentos.artigo.pareado "$(RODADA)" --processos "$(PROCESSOS)"

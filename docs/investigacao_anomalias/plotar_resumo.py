"""Regera a figura apenas a partir das metricas versionaveis deste diretorio."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
v=pd.read_csv(p/'metricas_validacao.csv')
d=pd.read_csv(p/'metricas_diagnostico.csv')
methods=[('gnn','GNN + IF'),('mlp','MLP + IF'),('features','Features + IF'),('contagem','Contagem'),('relacional','Relacional ajustado')]
fig,ax=plt.subplots(figsize=(11,5.4),layout='constrained')
x=np.arange(3);width=.16
for i,(method,label) in enumerate(methods):
    dev=pd.read_csv(p/'metricas_piloto_v2.csv')
    if method in ('contagem','relacional'):
        name={'contagem':'contagem_coparticipacao','relacional':'surpresa_coparticipacao'}[method]
        first=d.query("scenario=='injetado' and method==@name").top100.mean()
    else:first=dev.query("scenario=='injetado' and universe=='completo' and model==@method").recovered_100.mean()
    values=[first]+[v.query("scenario=='injetado' and universe=='completo' and model==@method and realizacao==@r").recovered_100.mean() for r in ['piloto_v2_8201_8202','piloto_v2_8301_8302']]
    bars=ax.bar(x+(i-2)*width,values,width,label=label)
    ax.bar_label(bars,labels=[f'{a:g}' for a in values],padding=3,fontsize=9)
ax.set_xticks(x,['Desenvolvimento\n8101/8102','Validação congelada 1\n8201/8202','Validação congelada 2\n8301/8302'])
ax.set_ylim(0,50);ax.set_ylabel('Alvos artificiais entre os 100 primeiros (de 50)')
ax.set_title('Recuperação de coparticipação artificial — 5.000 carteiras')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(ncols=3,loc='upper center',fontsize=9)
fig.suptitle('GNN/MLP: média das seeds 42, 73 e 2026; mesmo valor em cada seed. Não avalia fraude.',fontsize=9,y=.01)
fig.savefig(p/'recuperacao.png',dpi=170,bbox_inches='tight');fig.savefig(p/'recuperacao.pdf',bbox_inches='tight')

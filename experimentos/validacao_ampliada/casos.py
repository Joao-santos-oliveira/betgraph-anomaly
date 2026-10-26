"""Fichas descritivas e pareamento; nao atribui rotulos de fraude."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .executar import ROOT,OUT,OLD,FORWARD


def parear(cases,raw,caliper=.75):
    raw=raw.set_index('wallet_id').sort_index()
    z=StandardScaler().fit_transform(np.log1p(raw[['n_mercados_distintos','volume_total_usd']]))
    z=pd.DataFrame(z,index=raw.index,columns=['grau_z','volume_z'])
    available=set(raw.index)-set(cases);rows=[]
    for uid in sorted(cases):
        pool=z.loc[sorted(available)]
        delta=pool-z.loc[uid];ok=delta.abs().max(axis=1)<=caliper
        distances=(delta.loc[ok]**2).sum(axis=1).sort_values(kind='stable')
        if distances.empty:rows.append({'caso':uid,'controle':None,'distancia':None});continue
        control=int(distances.index[0]);available.remove(control)
        rows.append({'caso':uid,'controle':control,'distancia':float(np.sqrt(distances.iloc[0])),'delta_grau_z':float(delta.loc[control,'grau_z']),'delta_volume_z':float(delta.loc[control,'volume_z'])})
    return pd.DataFrame(rows)


def main():
    dest=OUT/'casos_reais';dest.mkdir(exist_ok=False)
    raw=pd.read_csv(OLD/'resultados/features_contexto.csv')
    rel=pd.read_csv(ROOT/'data/processed/investigacao_anomalias_v1/ranking_original/ranking.csv')
    rankings={'relacional':rel.set_index('wallet_id').posicao,'features':pd.read_csv(OLD/'resultados/features/ranking.csv').set_index('wallet_id').posicao}
    for model in ('gnn','mlp'):
        rankings[model]=pd.concat([pd.read_csv(OLD/f'resultados/{model}_seed_{seed}/ranking/ranking.csv').set_index('wallet_id').posicao for seed in (42,73,2026)],axis=1).median(axis=1)
    table=pd.DataFrame(rankings).sort_index();cases=set()
    for method in table:
        cases.update(table.sort_values(method,kind='stable').head(10).index)
    matches=parear(cases,raw);matches.to_csv(dest/'pareamento.csv',index=False)
    controls=set(matches.controle.dropna().astype(int));pool=sorted(set(raw.wallet_id)-cases-controls)
    random=set(map(int,np.random.default_rng(9401).choice(pool,10,replace=False)))
    selected=sorted(cases|controls|random);np.random.default_rng(9402).shuffle(selected)
    key=pd.DataFrame({'ficha':[f'F{i+1:03d}' for i in range(len(selected))],'wallet_id':selected})
    key['tipo']=['caso' if u in cases else 'controle' if u in controls else 'aleatoria' for u in selected]
    key=key.merge(table.reset_index(),on='wallet_id');key.to_csv(dest/'chave_nao_mostrar_revisores.csv',index=False)
    forms=pd.DataFrame({'ficha':key.ficha,'revisor':'','padrao_observado':'','explicacao_legitima':'','evidencia_insuficiente':'','confianca':'','observacoes':''})
    forms.to_csv(dest/'formulario_revisor_1.csv',index=False);forms.to_csv(dest/'formulario_revisor_2.csv',index=False)
    split=torch.load(OLD/'resultados/divisao.pt',weights_only=False,map_location='cpu');g=split['context'];pairs=g[FORWARD].edge_index.numpy()
    ids=g['user'].original_id.numpy();eids=g['event'].original_id.numpy();ui=dict(zip(ids,range(len(ids))))
    events=pd.read_csv(OLD/'entrada/event.csv').set_index('condition_node_id');r=rel.set_index('wallet_id');raw=raw.set_index('wallet_id')
    neighbors={int(u):set(eids[pairs[1,pairs[0]==i]]) for u,i in ui.items()}
    facts=[]
    for row in key.itertuples():
        uid=row.wallet_id;f=raw.loc[uid];markets=neighbors[uid];partner=r.loc[uid,'parceiro_wallet_id']
        common=sorted(markets & neighbors.get(int(partner),set())) if pd.notna(partner) else []
        cats=events.loc[sorted(markets)].categoria_principal.value_counts() if markets else pd.Series(dtype=int)
        facts.append({'ficha':row.ficha,'wallet_id':uid,'grau':len(markets),'volume':float(f.volume_total_usd),'compartilhados_maior_parceiro':len(common),'categoria_predominante':cats.index[0] if len(cats) else None,'fracao_categoria_predominante':float(cats.iloc[0]/len(markets)) if markets else None})
        lines=[f'# Ficha {row.ficha}', '',f'Carteira {uid}; mercados de contexto: {len(markets)}; volume de contexto: US$ {f.volume_total_usd:.2f}.',f'Parceiro de maior surpresa: {int(partner) if pd.notna(partner) else "ausente"}; mercados compartilhados: {len(common)}.', '', 'A seleção do parceiro usa o detector relacional para todas as fichas; o método que selecionou a carteira e seu score não são exibidos. Isso não torna a avaliação totalmente cega à estrutura.', '', '## Mercados compartilhados (até 10)', '']
        for eid in common[:10]:lines.append(f'- Mercado {eid}: {events.loc[eid,"question"]} — categoria {events.loc[eid,"categoria_principal"]}.')
        lines += ['',f'Exibidos {min(10,len(common))} de {len(common)} mercados compartilhados; não é a vizinhança completa.', '', '```mermaid','graph LR',f'  U["Carteira {uid}"]']
        if pd.notna(partner):
            lines.append(f'  V["Carteira {int(partner)}"]')
            for eid in common[:10]:lines.extend([f'  U --- M{eid}["Mercado {eid}"]',f'  V --- M{eid}'])
        lines+=['```','','Avaliar: padrão observado, explicações legítimas, informação insuficiente e confiança. Compartilhamento não demonstra coordenação, pessoa comum ou fraude.']
        (dest/f'{row.ficha}.md').write_text('\n'.join(lines))
    pd.DataFrame(facts).to_csv(dest/'inspecao_descritiva.csv',index=False)
    (dest/'README.md').write_text(f'# Inspeção de casos reais\n\n{len(cases)} casos (união top10), {len(controls)} controles pareados, 10 aleatórios. {matches.controle.isna().sum()} casos sem controle dentro do limite; não relaxamos o pareamento.\n\nEntregar apenas fichas F*.md e um formulário por revisor, sem chave ou pareamento. A avaliação humana está pendente; os dados descritivos não são rótulos. Não acessar a chave antes de preencher. Os subgrafos são limitados a10mercados compartilhados com um parceiro; distâncias visuais não representam embeddings.\n')
    print('Casos:',len(cases),'controles:',len(controls),'aleatorios:',len(random),flush=True)

if __name__=='__main__':main()

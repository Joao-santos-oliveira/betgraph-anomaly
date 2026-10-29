"""Fixtures artificiais: nenhuma avaliação das realizações de desenvolvimento."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experimentos.gnn_v3.dados import FWD,catalog,context_rows,intervene,prepare,corrupt,validate_change
from experimentos.gnn_v3.modelos import Encoder,graph,evaluation_examples,auxiliary,train
from experimentos.gnn_v3.scores import manual,pair_score,metrics,ranking

def fixture():
    nu,ne=400,80;g=HeteroData();g['user'].x=torch.zeros(nu,1);g['event'].x=torch.zeros(ne,1)
    g['user'].original_id=torch.arange(nu)+1000;g['event'].original_id=torch.arange(ne)+9000
    pairs=torch.tensor([(u,(u*7+j)%ne) for u in range(nu) for j in range(5)]).T
    g[FWD].edge_index=pairs;g['event','rev_bets_on','user'].edge_index=pairs.flip(0)
    # Use the actual reverse relation expected by legacy validators.
    from preparar_gnn import REVERSE
    if REVERSE!=('event','rev_bets_on','user'):
        del g['event','rev_bets_on','user'];g[REVERSE].edge_index=pairs.flip(0)
    split={'context':g}
    for k,j in [('train_positive',8),('val_positive',9),('test_positive',10)]:split[k]=torch.tensor([(u,(u*7+j)%ne) for u in range(nu)]).T
    split['all_positive']=torch.cat([pairs]+[split[k] for k in ('train_positive','val_positive','test_positive')],1)
    users=pd.DataFrame({'wallet_id':np.arange(nu)+1000,'wallet':[f'w{i}' for i in range(nu)]})
    events=pd.DataFrame({'condition_node_id':np.arange(ne)+9000,'duracao_mercado_segundos':np.arange(ne)+100.,'duracao_mercado_valida':True,'categoria_principal':'Sports'})
    rows=[]
    for u,m in split['all_positive'].T.tolist():
        for s in (0,1):
            rows.append({'wallet_id':u+1000,'wallet':f'w{u}','condition_node_id':m+9000,'condition_id':f'm{m}','token_outcome_id':s,'token_outcome':['Yes','No'][s],'volume_usd':float(100+u+s),'n_trades':2,'preco_medio':.4+.2*s,'ticket_medio':float(100+u+s)/2,'primeiro_trade':(pd.Timestamp('2026-09-16',tz='UTC')+pd.Timedelta(seconds=(u*133+m*29+s*700)%86400)).isoformat(),'duracao_trade_segundos':0.})
    return split,users,events,pd.DataFrame(rows)

class V3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.s,cls.u,cls.e,cls.a=fixture();cls.cat=catalog(cls.s,cls.a)
    def test_interventions_preserve_degrees_reserved_and_attributes(self):
        for scenario in ('S1','S2','S3_estrutura','S3_volume'):
            with self.subTest(scenario=scenario):
                s,a,sw,l,d,ledger=intervene(self.s,self.u,self.a,scenario,30101,self.cat)
                self.assertEqual(l.alvo.sum(),50);self.assertTrue(d['integridade']['graus_preservados'])
                self.assertTrue(np.allclose(a.ticket_medio,a.volume_usd/a.n_trades))
                if scenario in ('S1','S2'):
                    after=ledger.loc[ledger.estado.eq('depois')]
                    self.assertTrue(after.duracao_trade_segundos.eq(0).all())
                    if scenario=='S1':self.assertTrue(after.token_outcome_id.eq(0).all())
                    else:
                        for _,rows in after.groupby('condition_node_id'):
                            counts=rows.token_outcome_id.value_counts().reindex([0,1],fill_value=0)
                            self.assertLessEqual(abs(counts.iloc[0]-counts.iloc[1]),1)
                if scenario=='S3_volume':self.assertTrue(torch.equal(s['context'][FWD].edge_index,self.s['context'][FWD].edge_index))
        with self.assertRaises(AssertionError):intervene(self.s,self.u,self.a,'S1',30301,self.cat)
    def test_no_flags_or_reserved_attributes_in_inputs(self):
        base,x,t,_,_=prepare(self.s,self.u,self.e,self.a,self.cat)
        a=self.a.copy();a['alvo']=1;a['doadora']=True;a['grupo']='segredo'
        hidden=a.index.difference(context_rows(self.s,a).index);a.loc[hidden,['volume_usd','preco_medio']]=987654.
        u=self.u.assign(alvo=1);e=self.e.assign(alvo=1)
        _,x2,t2,_,_=prepare(self.s,u,e,a,self.cat)
        for k in x:self.assertTrue(torch.equal(x[k],x2[k]))
        for k in t:self.assertTrue(torch.equal(t[k],t2[k]))
        self.assertEqual(set(t),{'pairs','slots','attrs','binary'})
    def test_edges_and_outcomes_affect_gnn_but_not_mlp(self):
        _,x,t,_,_=prepare(self.s,self.u,self.e,self.a,self.cat);g=graph(x,t)
        tc=corrupt(t,56);self.assertTrue(torch.equal(tc['attrs'],corrupt(t,56)['attrs']))
        self.assertTrue(torch.equal(tc['pairs'],t['pairs']));self.assertTrue(torch.equal(tc['slots'],1-t['slots']))
        torch.manual_seed(42)
        for kind in ('C','D'):
            model=Encoder({k:v.shape[1] for k,v in x.items()},kind).eval();z=model(g)
            for field in ('attrs','slots'):
                changed={k:v.clone() for k,v in t.items()};changed[field]=tc[field];zc=model(graph(x,changed))
                self.assertEqual(torch.allclose(z['user'],zc['user']),kind=='D')
            model.train();z=model(g);zc=model(graph(x,tc));loss=auxiliary(model,z,zc,t,tc,torch.arange(20));loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
    def test_evaluation_shared_and_no_test_forward(self):
        from experimentos.gnn_v2.treino import treinar
        from experimentos.gnn_v2.features import enriquecer
        base,x,t,_,_=prepare(self.s,self.u,self.e,self.a,self.cat);vp,vl,_=evaluation_examples(base)
        for seed in (42,73,2026):
            torch.manual_seed(seed);p,l,_=evaluation_examples(base);self.assertTrue(torch.equal(vp,p));self.assertTrue(torch.equal(vl,l))
        with tempfile.TemporaryDirectory() as tmp:
            # Reference runs zero epochs on a fixture: checks exact legacy evaluation construction.
            enriched,_,_=enriquecer(base);_,r=treinar(enriched,Path(tmp)/'E','gnn',42,True,epochs=0)
            v=torch.load(Path(tmp)/'E/validacao.pt',weights_only=False);self.assertTrue(torch.equal(v['pairs'],vp));self.assertTrue(torch.equal(v['labels'],vl));self.assertFalse(r['teste_avaliado'])
        all_test=set(map(tuple,base['test_positive'].T.tolist()));self.assertFalse(all_test.intersection(map(tuple,vp.T.tolist())))
    def test_training_checkpoint_pipeline_on_tiny_fixture(self):
        from test_piloto_sintetico import fixture as tiny
        s,u,e,a,_=tiny();a['token_outcome']=a.token_outcome_id.astype(str);a['ticket_medio']=a.volume_usd/a.n_trades;a['duracao_trade_segundos']=0.
        base,x,t,_,_=prepare(s,u,e,a,catalog(s,a));vp,vl,_=evaluation_examples(base)
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('C','D'):
                out=Path(tmp)/kind;z,r=train(base,x,t,out,kind,42)
                self.assertFalse(r['teste_avaliado']);self.assertGreater(r['epochs_run'],0)
                v=torch.load(out/'validacao.pt',weights_only=False);self.assertTrue(torch.equal(v['pairs'],vp));self.assertTrue(torch.equal(v['labels'],vl))
                ck=torch.load(out/'melhor_modelo.pt',weights_only=False);m=Encoder(ck['dims'],kind);m.load_state_dict(ck['model_state']);m.eval()
                with torch.no_grad():again=m(graph(x,t))['user'].numpy()
                self.assertTrue(np.array_equal(z,again));self.assertTrue(np.isfinite(pair_score(base,z).score_anomalia).all())

    def test_mixed_timestamp_formats(self):
        from experimentos.gnn_v3.dados import seconds
        values=pd.Series(['2026-09-16 01:00:00+00:00','2026-09-16T01:00:00.500000+00:00'])
        self.assertTrue(np.array_equal(seconds(values),[3600.,3600.5]))

    def test_manual_and_metric_universes(self):
        base,_,_,_,_=prepare(self.s,self.u,self.e,self.a,self.cat);r=manual(base,self.a)
        self.assertEqual(set(r),{'A','B','O','T','V'})
        for a in r.values():self.assertEqual(len(a),400);self.assertTrue(np.isfinite(a.score_anomalia).all())
        labs=pd.DataFrame({'wallet_id':np.arange(400),'alvo':np.arange(400)<50,'elegivel':True,'doadora':False})
        rows=metrics(ranking(np.arange(400),-np.arange(400)),labs)
        self.assertTrue(all(v['top100']==50 and v['AP']==1. for v in rows))
    def test_corrupted_reserved_or_outcome_is_rejected(self):
        s=copy.deepcopy(self.s);s['test_positive']=s['test_positive'].clone();s['test_positive'][0,0]=3
        with self.assertRaises(AssertionError):validate_change(self.s,s,self.a,self.a,self.cat)
        bad=self.a.copy();bad.loc[0,'token_outcome']='wrong'
        with self.assertRaises(AssertionError):validate_change(self.s,self.s,self.a,bad,self.cat)

if __name__=='__main__':unittest.main()

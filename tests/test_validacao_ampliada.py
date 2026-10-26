import unittest
import numpy as np
import pandas as pd
import torch
from test_piloto_sintetico import fixture
from experimentos.validacao_ampliada.executar import randomizar,volume_injection,FORWARD
from experimentos.validacao_ampliada.casos import parear

class AmpliadaTest(unittest.TestCase):
    def test_nulo_preserva_particoes_e_graus(self):
        split,*_=fixture();changed,n=randomizar(split,9501)
        self.assertGreater(n,0)
        for k in ('train_positive','val_positive','test_positive'):self.assertTrue(torch.equal(split[k],changed[k]))
        second,_=randomizar(split,9501)
        self.assertTrue(torch.equal(changed['context'][FORWARD].edge_index,second['context'][FORWARD].edge_index))
    def test_volume_altera_so_contexto_e_mantem_alvos(self):
        from torch_geometric.data import HeteroData
        g=HeteroData();g['user'].original_id=torch.arange(60);g['event'].original_id=torch.arange(7)
        pairs=torch.tensor([(u,e) for u in range(60) for e in range(5)]).T;g[FORWARD].edge_index=pairs
        edges=pd.DataFrame({'wallet_id':pairs[0].tolist()+[0],'condition_node_id':pairs[1].tolist()+[6],'volume_usd':1.,'n_trades':1,'preco_medio':.5})
        a,labels,_=volume_injection({'context':g},edges,9201,10)
        b,labels2,_=volume_injection({'context':g},edges,9201,100)
        self.assertEqual(labels.alvo.sum(),50);pd.testing.assert_frame_equal(labels,labels2)
        self.assertEqual(a.iloc[-1].volume_usd,1);self.assertEqual((a.volume_usd==10).sum(),250)
        pd.testing.assert_frame_equal(edges.drop(columns='volume_usd'),b.drop(columns='volume_usd'))
    def test_pareamento_sem_reposicao_e_sem_forcar_controle(self):
        raw=pd.DataFrame({'wallet_id':[1,2,3,4],'n_mercados_distintos':[5,5,50,50],'volume_total_usd':[10,10,100,100]})
        matched=parear({1,3},raw)
        self.assertEqual(set(matched.controle),{2,4})
        self.assertTrue(parear({1,2,3,4},raw).controle.isna().all())

if __name__=='__main__':unittest.main()

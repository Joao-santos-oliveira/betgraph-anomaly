import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from test_piloto_sintetico import fixture
from experimentos.gnn_v2.features import enriquecer,FORWARD
from experimentos.gnn_v2.detectores import hibrido,ranking
from experimentos.gnn_v2.objetivos import Contraste,corromper
from experimentos.gnn_v2.treino import treinar
from experimentos.investigacao_anomalias.relacional import scores_relacionais

class V2Test(unittest.TestCase):
    def test_features_nao_usam_particoes_reservadas(self):
        split,*_=fixture();before=copy.deepcopy(split)
        a,bundle,_=enriquecer(split)
        changed=copy.deepcopy(split);changed['test_positive']=changed['test_positive'].flip(1);changed['val_positive']=torch.zeros_like(changed['val_positive'])
        b,_,_=enriquecer(changed)
        self.assertTrue(torch.equal(a['context']['user'].x,b['context']['user'].x))
        self.assertTrue(torch.equal(split['context']['user'].x,before['context']['user'].x))
        for part in ['train_positive','val_positive','test_positive']:self.assertTrue(torch.equal(a[part],split[part]))
        self.assertEqual(a['context']['user'].x.shape[1],10)
    def test_hibrido_limites_e_reducao_ao_score_direto(self):
        p=np.array([[0,0,1,1,2,2],[0,1,0,1,2,3]])
        direct=scores_relacionais(p,4,4)['surpresa_coparticipacao']
        np.testing.assert_allclose(hibrido(p,4,4,np.ones((4,2))),direct,rtol=1e-12,atol=1e-12)
        z=np.array([[1.,0.],[-1.,0.],[1.,0.],[0.,0.]])
        np.testing.assert_allclose(hibrido(p,4,4,z),0)
        random=np.random.default_rng(42).normal(size=(4,3));scores=hibrido(p,4,4,random)
        self.assertTrue((scores<=direct+1e-12).all());self.assertTrue((scores>=0).all())
        self.assertEqual(ranking([9,2,7],[0.,0.,0.]).wallet_id.tolist(),[2,7,9])
    def test_corrupcao_reprodutivel_preserva_inativos_e_gradientes(self):
        x={k:torch.arange(24,dtype=torch.float).reshape(6,4) for k in ['user','event']}
        active={k:torch.tensor([True,True,True,True,False,False]) for k in x}
        a=corromper(x,active,torch.Generator().manual_seed(4));b=corromper(x,active,torch.Generator().manual_seed(4))
        for k in x:
            self.assertTrue(torch.equal(a[k],b[k]));self.assertTrue(torch.equal(a[k][4:],x[k][4:]));self.assertEqual(sorted(a[k][:4,0].tolist()),sorted(x[k][:4,0].tolist()))
        normal={k:torch.randn(6,4,requires_grad=True) for k in x};corrupt={k:torch.randn(6,4,requires_grad=True) for k in x}
        head=Contraste(4);loss=head(normal,corrupt,active);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
    def test_treino_contrastivo_checkpoint_reproduzivel(self):
        split,*_=fixture();split,_,_=enriquecer(split)
        with tempfile.TemporaryDirectory() as tmp:
            a,ma=treinar(split,Path(tmp)/'a','gnn',42,True,epochs=2)
            b,mb=treinar(split,Path(tmp)/'b','gnn',42,True,epochs=2)
            np.testing.assert_array_equal(a,b);self.assertFalse(ma['teste_avaliado']);self.assertEqual(ma['best_epoch'],mb['best_epoch'])
            saved=torch.load(Path(tmp)/'a/melhor_modelo.pt',weights_only=False)
            self.assertIsNotNone(saved['head_state']);self.assertTrue(np.isfinite(a).all())

if __name__=='__main__':unittest.main()

"""Propriedades do score e limite construtivo da agregacao media."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import poisson
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experimentos.investigacao_anomalias.relacional import scores_relacionais


class RelacionalTest(unittest.TestCase):
    def test_contagem_expectativa_simetria_e_isolados(self):
        pairs=np.array([[0,0,1,1,2,2,3],[0,1,0,1,2,3,2]],dtype=np.int64)
        stats=scores_relacionais(pairs,5,4)
        expected=-poisson.logsf(1,4/7)/np.log(10)/3
        np.testing.assert_allclose(stats['surpresa_coparticipacao'],[expected,expected,0,0,0])
        np.testing.assert_array_equal(stats['mercados_compartilhados'],[2,2,0,0,0])
        self.assertEqual(stats['melhor_parceiro_indice'][4],-1)
        np.testing.assert_allclose(stats['contagem_coparticipacao'],[2/3,2/3,0,0,0])

    def test_invariancia_ordem_indices_e_blocos(self):
        pairs=np.array([(u,e) for u in range(8) for e in (u//2*2,u//2*2+1)],dtype=np.int64).T
        before=scores_relacionais(pairs,9,8,block_size=1)
        reordered=scores_relacionais(pairs[:,::-1],9,8,block_size=4)
        for name in before: np.testing.assert_allclose(before[name],reordered[name])
        up=np.array([3,0,7,1,2,5,8,6,4]);ep=np.arange(8)[::-1]
        permuted=scores_relacionais(np.array([up[pairs[0]],ep[pairs[1]]]),9,8)
        for name in ('surpresa_coparticipacao','contagem_coparticipacao','grau_contexto'):
            np.testing.assert_allclose(before[name],permuted[name][up])

    def test_rejeita_duplicatas_e_endpoints(self):
        with self.assertRaisesRegex(ValueError,'duplicados'):
            scores_relacionais(np.array([[0,0],[1,1]]),2,2)
        with self.assertRaisesRegex(ValueError,'Endpoint'):
            scores_relacionais(np.array([[0],[2]]),2,2)
        self.assertTrue((scores_relacionais(np.empty((2,0),dtype=np.int64),2,2)['surpresa_coparticipacao']==0).all())

    def test_gnn_media_nao_distingue_motivos_com_features_constantes(self):
        import torch
        from modelo_gnn import GNNBipartida
        from preparar_gnn import FORWARD,REVERSE
        # Mesmos graus 2, todos os atributos iguais por tipo: blocos 2x2 vs ciclo longo.
        blocks=torch.tensor([(u,e) for u in range(8) for e in (u//2*2,u//2*2+1)]).T
        cycle=torch.tensor([(u,e) for u in range(8) for e in (u,(u+1)%8)]).T
        torch.manual_seed(42)
        model=GNNBipartida(2,3);model.eval()
        x={'user':torch.ones(8,2),'event':torch.ones(8,3)}
        with torch.no_grad():
            a=model(x,{FORWARD:blocks,REVERSE:blocks.flip(0)})
            b=model(x,{FORWARD:cycle,REVERSE:cycle.flip(0)})
        torch.testing.assert_close(a['user'],b['user'],rtol=0,atol=0)
        sa=scores_relacionais(blocks.numpy(),8,8)['surpresa_coparticipacao']
        sb=scores_relacionais(cycle.numpy(),8,8)['surpresa_coparticipacao']
        self.assertTrue((sa>0).all());self.assertTrue((sb==0).all())


if __name__=='__main__': unittest.main()

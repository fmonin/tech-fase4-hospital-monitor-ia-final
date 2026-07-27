"""
Garante que o esquema de atributos do treino alternativo com vídeos reais
(treinar_modelo_movimento_real.py) fica sempre igual ao do treino sintético
(conjunto_dados_movimento.py) — os dois alimentam o mesmo formato de
modelo, então não podem divergir silenciosamente.
"""

import numpy as np

from src.training import conjunto_dados_movimento
from src.training.treinar_modelo_movimento_real import COLUNAS_JANELA


def test_colunas_do_treino_real_batem_com_o_conjunto_sintetico():
    X, _ = conjunto_dados_movimento.construir_conjunto_dados(n_por_classe=5)
    assert set(X.columns) == set(COLUNAS_JANELA)


def test_extrair_atributos_da_janela_devolve_16_colunas():
    rng = np.random.default_rng(1)
    janela = conjunto_dados_movimento._gerar_janela_pontos(rng, conjunto_dados_movimento.TAMANHO_JANELA_FRAMES, anomala=False)
    atributos = conjunto_dados_movimento.extrair_atributos_da_janela(janela)
    assert len(atributos) == 16

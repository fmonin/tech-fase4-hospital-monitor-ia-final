"""
Testa o pipeline de treino em si — não só os modelos já treinados, mas o
processo de construir o conjunto de dados + treinar + avaliar. Roda rápido
porque usa poucos exemplos (o treino "de verdade" em src/training/treinar_*.py
usa mais exemplos, mas o código exercitado é exatamente o mesmo).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.training import conjunto_dados_audio, conjunto_dados_movimento, conjunto_dados_vitais
from src.training.utils import avaliar_modelo


def _treinar_rapido(X, y):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    modelo = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=1)
    modelo.fit(X_treino, y_treino)
    return avaliar_modelo(modelo, X_teste, y_teste, "teste")


def test_conjunto_dados_vitais_e_balanceado_e_separavel():
    X, y = conjunto_dados_vitais.construir_conjunto_dados(n_por_classe=60)
    assert len(X) == 120
    assert y.sum() == 60  # metade das amostras é anômala

    metricas = _treinar_rapido(X, y)
    assert metricas["acuracia"] > 0.85  # anomalias injetadas devem ser bem separáveis


def test_conjunto_dados_movimento_e_balanceado_e_separavel():
    X, y = conjunto_dados_movimento.construir_conjunto_dados(n_por_classe=60)
    assert len(X) == 120
    assert y.sum() == 60

    metricas = _treinar_rapido(X, y)
    assert metricas["acuracia"] > 0.85


def test_conjunto_dados_audio_e_balanceado_e_separavel():
    X, y = conjunto_dados_audio.construir_conjunto_dados(n_por_classe=60)
    assert len(X) == 120
    assert y.sum() == 60

    metricas = _treinar_rapido(X, y)
    assert metricas["acuracia"] > 0.85

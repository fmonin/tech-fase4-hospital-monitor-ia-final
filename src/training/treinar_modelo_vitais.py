"""
Passo a passo do treino do classificador de anomalias em sinais vitais.

Rodar diretamente:  python -m src.training.treinar_modelo_vitais

Etapas (propositalmente explícitas, para ficar fácil de acompanhar/explicar
numa apresentação):
  1. Construir o conjunto de dados rotulado (atributos + rótulo normal/anômalo).
  2. Separar treino e teste.
  3. Treinar um RandomForestClassifier (escolhido por ser simples de
     explicar, robusto a atributos em escalas diferentes e por permitir
     inspecionar quais sinais mais pesaram na decisão).
  4. Avaliar no conjunto de teste (dados que o modelo nunca viu).
  5. Salvar o modelo treinado em models/vitais_rf.joblib.
"""

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.training.conjunto_dados_vitais import construir_conjunto_dados
from src.training.utils import avaliar_modelo, salvar_modelo_treinado

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("treino_vitais")

NOME_MODELO = "vitais_rf"


def treinar():
    logger.info("1/5 - Construindo conjunto de dados rotulado (sintético, com anomalias injetadas)...")
    X, y = construir_conjunto_dados(n_por_classe=500)
    logger.info("Conjunto de dados construído: %d exemplos, %d atributos.", len(X), X.shape[1])

    logger.info("2/5 - Separando treino (80%%) e teste (20%%)...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    logger.info("3/5 - Treinando RandomForestClassifier...")
    modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    modelo.fit(X_treino, y_treino)

    logger.info("4/5 - Avaliando no conjunto de teste...")
    metricas = avaliar_modelo(modelo, X_teste, y_teste, NOME_MODELO)

    importancias = sorted(zip(X.columns, modelo.feature_importances_), key=lambda t: -t[1])
    logger.info("Atributos mais importantes: %s", [f"{nome} ({peso:.2f})" for nome, peso in importancias[:5]])

    logger.info("5/5 - Salvando modelo treinado...")
    caminho = salvar_modelo_treinado(modelo, NOME_MODELO, metricas, list(X.columns))

    print(f"\nModelo de sinais vitais treinado e salvo em: {caminho}")
    print(f"Acurácia no teste: {metricas['acuracia'] * 100:.1f}%")
    return modelo, metricas


if __name__ == "__main__":
    treinar()

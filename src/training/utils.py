"""
Funções compartilhadas pelos três scripts de treino (vitais, movimento,
áudio): avaliação do modelo e persistência em disco. Centralizar isso aqui
evita repetir a mesma lógica de "treinar, avaliar, salvar" três vezes.
"""

import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

PASTA_MODELOS = Path(__file__).resolve().parent.parent.parent / "models"
PASTA_MODELOS.mkdir(exist_ok=True)


def avaliar_modelo(modelo, X_teste, y_teste, nome_modelo: str) -> dict:
    """Calcula as métricas padrão de classificação e devolve tudo num dicionário."""
    y_previsto = modelo.predict(X_teste)

    metricas = {
        "modelo": nome_modelo,
        "acuracia": round(accuracy_score(y_teste, y_previsto), 4),
        "precisao": round(precision_score(y_teste, y_previsto), 4),
        "recall": round(recall_score(y_teste, y_previsto), 4),
        "f1_score": round(f1_score(y_teste, y_previsto), 4),
        "matriz_confusao": confusion_matrix(y_teste, y_previsto).tolist(),
        "relatorio_completo": classification_report(y_teste, y_previsto, target_names=["normal", "anomalo"]),
    }

    logger.info("=== Avaliação: %s ===", nome_modelo)
    logger.info("Acurácia: %.2f%% | Precisão: %.2f%% | Recall: %.2f%% | F1: %.2f%%",
                 metricas["acuracia"] * 100, metricas["precisao"] * 100,
                 metricas["recall"] * 100, metricas["f1_score"] * 100)
    logger.info("Matriz de confusão (linhas=real, colunas=previsto):\n%s", metricas["matriz_confusao"])

    if metricas["acuracia"] >= 0.98:
        logger.warning(
            "Métrica muito alta (>=98%%). Em dados sintéticos isso pode indicar separação artificial; "
            "avalie viés, sobreajuste e valide em dados reais antes de uso clínico."
        )

    return metricas


def salvar_modelo_treinado(modelo, nome_arquivo: str, metricas: dict, colunas_atributos: list[str]) -> Path:
    """Salva o modelo treinado (.joblib) e suas métricas/atributos (.json) em models/."""
    caminho_modelo = PASTA_MODELOS / f"{nome_arquivo}.joblib"
    caminho_metricas = PASTA_MODELOS / f"{nome_arquivo}_metricas.json"

    joblib.dump(modelo, caminho_modelo)

    metricas_para_salvar = {k: v for k, v in metricas.items() if k != "relatorio_completo"}
    metricas_para_salvar["colunas_atributos"] = colunas_atributos
    caminho_metricas.write_text(json.dumps(metricas_para_salvar, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Modelo salvo em %s", caminho_modelo)
    return caminho_modelo


def carregar_modelo_treinado(nome_arquivo: str):
    """
    Tenta carregar um modelo já treinado. Devolve None se ele ainda não
    existe (ex: usuário não rodou o script de treino) — quem chama decide
    o fallback, em vez de o programa quebrar.
    """
    caminho_modelo = PASTA_MODELOS / f"{nome_arquivo}.joblib"
    if not caminho_modelo.exists():
        return None
    return joblib.load(caminho_modelo)


def carregar_colunas_atributos(nome_arquivo: str) -> list[str] | None:
    caminho_metricas = PASTA_MODELOS / f"{nome_arquivo}_metricas.json"
    if not caminho_metricas.exists():
        return None
    return json.loads(caminho_metricas.read_text(encoding="utf-8")).get("colunas_atributos")

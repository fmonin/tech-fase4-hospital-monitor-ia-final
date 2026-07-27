"""
Classifica o risco de fadiga/disartria na voz usando o modelo treinado em
src/training/treinar_modelo_audio.py, a partir das características
acústicas já extraídas por src/audio/voice_features.py.

Se o modelo treinado não existir em disco, devolve None — quem chama
(src/audio/audio_pipeline.py) já sabe usar o indicador heurístico como
alternativa nesse caso.
"""

import logging

import pandas as pd

from src.training.utils import carregar_modelo_treinado

logger = logging.getLogger(__name__)

NOME_MODELO = "audio_rf"


def classificar_risco_fala(indicadores_acusticos: dict) -> dict | None:
    """
    Recebe o dicionário devolvido por `extrair_indicadores_de_fala` e
    devolve a classificação do modelo treinado: {"classe": "normal"|"alterado",
    "confianca": 0.0-1.0}. Devolve None se o modelo ainda não foi treinado.
    """
    modelo = carregar_modelo_treinado(NOME_MODELO)
    if modelo is None:
        logger.warning(
            "Modelo treinado '%s' não encontrado em models/. "
            "Rode: python -m src.training.treinar_modelo_audio", NOME_MODELO,
        )
        return None

    atributos = pd.DataFrame([{
        "variabilidade_pitch_hz": indicadores_acusticos["variabilidade_pitch_hz"],
        "proporcao_silencio": indicadores_acusticos["proporcao_silencio"],
        "energia_media": indicadores_acusticos["energia_media"],
    }])

    previsto = modelo.predict(atributos)[0]
    probabilidade = modelo.predict_proba(atributos)[0][1]

    return {
        "classe": "alterado" if previsto == 1 else "normal",
        "confianca": round(float(probabilidade), 3),
    }

"""
Conjunto de dados rotulado para treinar o classificador de risco na voz
(indícios de fadiga/disartria), a partir das MESMAS três características
acústicas já calculadas em src/audio/voice_features.py: variabilidade de
pitch, proporção de silêncio e energia média.

Em vez de gerar e analisar milhares de arquivos .wav (lento, e o resultado
seria estatisticamente idêntico), simulamos diretamente a distribuição
dessas três características para voz "normal" e voz com "indício de
alteração" — é a mesma ideia usada nos outros dois conjuntos de dados, só
que direto no espaço de atributos em vez de no áudio bruto.
"""

import numpy as np
import pandas as pd


def _gerar_exemplo(rng: np.random.Generator, alterado: bool) -> dict:
    if not alterado:
        # Voz "saudável": pitch com variação natural, poucas pausas, energia normal.
        return {
            "variabilidade_pitch_hz": rng.normal(24, 7),
            "proporcao_silencio": np.clip(rng.normal(0.20, 0.07), 0, 1),
            "energia_media": max(rng.normal(0.038, 0.012), 0.001),
        }

    # Voz com indício de fadiga/disartria: fala mais monótona (pitch pouco
    # variável), mais pausas, energia mais baixa.
    return {
        "variabilidade_pitch_hz": max(rng.normal(12, 5), 0),
        "proporcao_silencio": np.clip(rng.normal(0.36, 0.10), 0, 1),
        "energia_media": max(rng.normal(0.022, 0.010), 0.0001),
    }


def construir_conjunto_dados(n_por_classe: int = 500, semente: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Gera um conjunto balanceado: voz normal x voz com indício de alteração."""
    rng = np.random.default_rng(semente)
    linhas, rotulos = [], []

    for _ in range(n_por_classe):
        linhas.append(_gerar_exemplo(rng, alterado=False))
        rotulos.append(0)

    for _ in range(n_por_classe):
        linhas.append(_gerar_exemplo(rng, alterado=True))
        rotulos.append(1)

    X = pd.DataFrame(linhas)
    y = pd.Series(rotulos, name="alterado")

    # Pequeno ruído de rótulo para reduzir separação artificial perfeita.
    proporcao_ruido_rotulo = 0.04
    n_ruido = max(1, int(len(y) * proporcao_ruido_rotulo))
    indices_ruido = rng.choice(len(y), size=n_ruido, replace=False)
    y.iloc[indices_ruido] = 1 - y.iloc[indices_ruido]

    return X, y

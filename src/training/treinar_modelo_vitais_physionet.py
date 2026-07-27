"""
Treino ALTERNATIVO do modelo de sinais vitais, usando sinal real do
PhysioNet — complementar ao treino supervisionado com dados sintéticos de
`treinar_modelo_vitais.py` (que continua sendo o modelo usado pelo sistema
em produção, ver src/anomalies/vitals_anomaly.py).

Por que dois jeitos de treinar o mesmo tipo de modelo? Cada um mostra uma
técnica diferente, válida em cenários diferentes:

- `treinar_modelo_vitais.py` (RandomForest supervisionado): precisa de
  rótulo confiável, então usa dados 100% sintéticos com anomalia injetada.
  Métricas de acerto (acurácia/precisão/recall) são diretas de calcular e
  confiáveis, porque o gabarito é conhecido.
- Este script (IsolationForest não supervisionado): não exige rótulo, então
  consegue treinar em cima de um sinal REAL — a frequência cardíaca é
  derivada das anotações reais de batimento do MIT-BIH Arrhythmia Database
  (PhysioNet, aberto, sem necessidade de credencial). Pressão arterial e
  SpO2 auxiliares são simuladas de forma correlacionada com essa FC real,
  só para completar as 4 métricas que o resto do sistema espera — isso fica
  documentado aqui e no relatório técnico, sem esconder o que é real e o
  que não é.

Rodar:  python -m src.training.treinar_modelo_vitais_physionet

Requer acesso à internet a physionet.org (baixa o registro "100" da base
"mitdb" via `wfdb`). Se não houver rede disponível, o script falha com uma
mensagem clara — ele não tem fallback sintético, porque todo o propósito
aqui é justamente usar sinal real.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.training.utils import PASTA_MODELOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("treino_vitais_physionet")

NOME_MODELO = "vitais_isolationforest_physionet"
COLUNAS_SINAIS = ["frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"]


def construir_serie_real(registro: str = "100", maximo_amostras: int = 1800) -> pd.DataFrame:
    """
    Baixa o registro de ECG anotado do MIT-BIH e deriva a frequência
    cardíaca real a partir do intervalo entre batimentos (R-R). SpO2 e
    pressão são simuladas de forma correlacionada com essa FC real, só
    para completar as features — não são medidas reais desse paciente.
    """
    import wfdb

    logger.info("Baixando registro real '%s' da base 'mitdb' no PhysioNet...", registro)
    try:
        anotacao = wfdb.rdann(registro, "atr", pn_dir="mitdb")
        gravacao = wfdb.rdrecord(registro, pn_dir="mitdb")
    except Exception as erro:
        raise ConnectionError(
            f"Não foi possível baixar o registro '{registro}' do PhysioNet (mitdb). "
            f"Verifique a conexão com a internet. Detalhe: {erro}"
        ) from erro

    frequencia_amostragem = float(gravacao.fs)
    instantes_batimento = np.asarray(anotacao.sample) / frequencia_amostragem
    intervalos_rr = np.diff(instantes_batimento)

    frequencia_cardiaca = 60.0 / intervalos_rr
    frequencia_cardiaca = frequencia_cardiaca[(frequencia_cardiaca > 30) & (frequencia_cardiaca < 220)]
    frequencia_cardiaca = frequencia_cardiaca[:maximo_amostras]

    rng = np.random.default_rng(42)
    n = len(frequencia_cardiaca)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "frequencia_cardiaca": frequencia_cardiaca,
        "spo2": np.clip(97 - np.maximum(frequencia_cardiaca - 120, 0) * 0.015 + rng.normal(0, 0.5, n), 85, 100),
        "pressao_sistolica": 120 + (frequencia_cardiaca - 75) * 0.15 + rng.normal(0, 6, n),
        "pressao_diastolica": 78 + (frequencia_cardiaca - 75) * 0.08 + rng.normal(0, 4, n),
    }).round(2)

    logger.info("Série real construída: %d amostras (FC real do MIT-BIH; SpO2/pressão simuladas).", n)
    return df


def treinar():
    logger.info("1/4 - Baixando e derivando série real do PhysioNet (MIT-BIH)...")
    df = construir_serie_real()

    logger.info("2/4 - Treinando IsolationForest (não supervisionado, sem rótulo)...")
    modelo = IsolationForest(n_estimators=250, contamination=0.04, random_state=42)
    modelo.fit(df[COLUNAS_SINAIS])

    logger.info("3/4 - Aplicando o modelo sobre a própria série para inspecionar o resultado...")
    previsao = modelo.predict(df[COLUNAS_SINAIS])
    n_anomalias = int((previsao == -1).sum())
    logger.info("Anomalias identificadas na série real: %d de %d amostras (%.1f%%).",
                n_anomalias, len(df), 100 * n_anomalias / len(df))

    logger.info("4/4 - Salvando modelo e série usada...")
    import joblib
    caminho_modelo = PASTA_MODELOS / f"{NOME_MODELO}.joblib"
    joblib.dump(modelo, caminho_modelo)
    df.to_csv(PASTA_MODELOS / f"{NOME_MODELO}_serie_treino.csv", index=False)

    print(f"\nModelo (IsolationForest, sinal real do PhysioNet) salvo em: {caminho_modelo}")
    print(f"Anomalias identificadas na própria série de treino: {n_anomalias}/{len(df)}")
    return modelo, {"n_amostras": len(df), "n_anomalias": n_anomalias}


if __name__ == "__main__":
    treinar()

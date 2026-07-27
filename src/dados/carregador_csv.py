"""
Carregamento de sinais vitais a partir de um arquivo CSV externo.

Alguns dados clínicos não vêm de uma API (VitalDB, PhysioNet) nem de um
gerador sintético — vêm de uma planilha/exportação que o hospital já tem.
Este módulo cobre esse caso: lê qualquer CSV com colunas de sinais vitais
(em português ou inglês, ver `MAPA_COLUNAS` abaixo) e devolve o mesmo
formato usado pelo resto do sistema.

O arquivo `amostras/sinais_vitais_exemplo.csv` incluído no projeto é um
exemplo real de uso: um CSV pequeno, com uma anomalia clara na quarta linha
(frequência cardíaca de 145 bpm e SpO2 de 89%), útil para testar o
carregador e a detecção de limites clínicos sem precisar gerar nada.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Aceita tanto os nomes em português usados internamente quanto variações
# comuns em inglês (ex: exportações de outros sistemas/planilhas).
MAPA_COLUNAS = {
    "heart_rate": "frequencia_cardiaca",
    "hr": "frequencia_cardiaca",
    "frequencia_cardiaca": "frequencia_cardiaca",
    "spo2": "spo2",
    "oxygen_saturation": "spo2",
    "systolic_bp": "pressao_sistolica",
    "pressao_sistolica": "pressao_sistolica",
    "diastolic_bp": "pressao_diastolica",
    "pressao_diastolica": "pressao_diastolica",
    "timestamp": "timestamp",
    "data": "timestamp",
}

COLUNAS_OBRIGATORIAS = ["timestamp", "frequencia_cardiaca"]


def carregar_sinais_vitais_de_csv(caminho_csv: str) -> pd.DataFrame:
    """
    Lê um CSV de sinais vitais e devolve um DataFrame no formato padrão do
    projeto: timestamp | frequencia_cardiaca | spo2 | pressao_sistolica |
    pressao_diastolica (as três últimas só aparecem se estiverem no arquivo).
    """
    df = pd.read_csv(caminho_csv)

    colunas_renomeadas = {
        coluna: MAPA_COLUNAS[coluna.strip().lower()]
        for coluna in df.columns
        if coluna.strip().lower() in MAPA_COLUNAS
    }
    df = df.rename(columns=colunas_renomeadas)

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(
            f"O CSV precisa ter as colunas {COLUNAS_OBRIGATORIAS} (ou equivalentes em inglês, "
            f"ver MAPA_COLUNAS). Colunas faltando: {faltando}. Colunas encontradas: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("CSV de sinais vitais carregado: %s (%d amostras).", caminho_csv, len(df))
    return df

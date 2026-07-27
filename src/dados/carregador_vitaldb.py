"""
Carregamento de um caso REAL do VitalDB — banco aberto de sinais vitais de
cirurgias reais (anonimizadas), disponível em:

    https://physionet.org/content/vitaldb/1.0.0/   (via PhysioNet)
    https://vitaldb.net/                             (API oficial, usada aqui)

Usamos aqui a API oficial (biblioteca `vitaldb`, mais simples que baixar o
.zip completo do PhysioNet): ela busca só as faixas de sinal que
precisamos, de um único caso, direto do servidor da VitalDB.

Estes dados são usados para VALIDAÇÃO/DEMONSTRAÇÃO do modelo já treinado
(ver README e docs/relatorio_tecnico.md para o porquê de não treinarmos
diretamente neles: não há rótulo de anomalia pronto). Assim como no
carregador_physionet.py, qualquer falha de rede levanta ConnectionError —
quem chama decide o fallback.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Faixas de sinal do VitalDB (monitor Solar8000, o mais comum na base).
FAIXA_FC = "Solar8000/HR"
FAIXA_SPO2 = "Solar8000/PLETH_SPO2"
FAIXA_PA_SISTOLICA = "Solar8000/NIBP_SBP"
FAIXA_PA_DIASTOLICA = "Solar8000/NIBP_DBP"


def carregar_caso_real(caso_id: int = 1, intervalo_segundos: int = 60) -> pd.DataFrame:
    """
    Baixa um caso real do VitalDB e devolve um DataFrame no mesmo formato
    usado pelo resto do projeto:

        timestamp | frequencia_cardiaca | spo2 | pressao_sistolica | pressao_diastolica

    `intervalo_segundos=60` reamostra os sinais para 1 leitura por minuto,
    consistente com o resto do pipeline (mesma granularidade dos dados
    sintéticos e do BIDMC).
    """
    import vitaldb  # import local: só necessário quando esta função é usada

    faixas = [FAIXA_FC, FAIXA_SPO2, FAIXA_PA_SISTOLICA, FAIXA_PA_DIASTOLICA]
    logger.info("Baixando caso real #%d do VitalDB...", caso_id)

    try:
        valores = vitaldb.load_case(caso_id, faixas, intervalo_segundos)
    except Exception as erro:
        raise ConnectionError(
            f"Não foi possível baixar o caso {caso_id} do VitalDB. "
            f"Verifique a conexão com a internet. Detalhe: {erro}"
        ) from erro

    df = pd.DataFrame(valores, columns=["frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"])
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq=f"{intervalo_segundos}s")
    df = df[["timestamp", "frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"]]

    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ConnectionError(f"Caso {caso_id} do VitalDB veio vazio (sem sobreposição entre as faixas pedidas).")

    logger.info("Caso real #%d carregado: %d amostras.", caso_id, len(df))
    return df

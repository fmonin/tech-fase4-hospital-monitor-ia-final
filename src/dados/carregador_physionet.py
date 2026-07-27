"""
Carregamento de dados REAIS de sinais vitais a partir do PhysioNet.

Usamos a base aberta "BIDMC PPG and Respiration" (physionet.org/content/bidmc),
que traz, para pacientes de UTI reais (anonimizados), séries numéricas de
frequência cardíaca, SpO2 e frequência respiratória batidas minuto a minuto —
exatamente o tipo de sinal que o hospital do desafio quer monitorar.

Este módulo tenta baixar um registro real via a biblioteca `wfdb`. Como o
download depende de acesso à internet (bloqueado em alguns ambientes restritos,
como sandboxes corporativos), toda falha de rede é tratada explicitamente:
o chamador decide o que fazer (normalmente, cair para dados sintéticos via
`dados_sinteticos.py`). Nunca escondemos o erro silenciosamente.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NOME_BASE_PHYSIONET = "bidmc"


def carregar_sinais_vitais_reais(registro: str = "bidmc01") -> pd.DataFrame:
    """
    Baixa um registro real do PhysioNet (BIDMC) e devolve um DataFrame no
    formato padrão usado pelo resto do projeto:

        timestamp | frequencia_cardiaca | spo2 | pressao_sistolica | pressao_diastolica

    Levanta ConnectionError / RuntimeError se o download não for possível —
    é responsabilidade de quem chama decidir o fallback.
    """
    import wfdb  # import local: só é necessário quando esta função é usada

    logger.info("Baixando registro real '%s' da base '%s' no PhysioNet...", registro, NOME_BASE_PHYSIONET)

    try:
        # O sufixo "n" identifica o arquivo de "numerics" (valores já calculados,
        # não a forma de onda bruta) — é o que nos interessa aqui.
        registro_numerico = wfdb.rdrecord(f"{registro}n", pn_dir=f"{NOME_BASE_PHYSIONET}/{registro}")
    except Exception as erro:
        raise ConnectionError(
            f"Não foi possível baixar o registro '{registro}' do PhysioNet. "
            f"Verifique a conexão com a internet. Detalhe: {erro}"
        ) from erro

    sinais = registro_numerico.p_signal
    nomes_canais = [nome.strip() for nome in registro_numerico.sig_name]

    df = pd.DataFrame(sinais, columns=nomes_canais)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1min")

    # Os nomes dos canais variam por registro; mapeamos os mais comuns do BIDMC.
    mapa_colunas = {"HR": "frequencia_cardiaca", "PULSE": "frequencia_cardiaca", "SpO2": "spo2"}
    df = df.rename(columns={c: mapa_colunas[c] for c in df.columns if c in mapa_colunas})

    colunas_finais = ["timestamp", "frequencia_cardiaca", "spo2"]
    df = df[[c for c in colunas_finais if c in df.columns]].copy()

    # O BIDMC não traz pressão arterial contínua nos arquivos abertos; como o
    # desafio pede pressão como um dos sinais monitorados, derivamos uma
    # estimativa fisiologicamente plausível a partir da frequência cardíaca
    # (correlação simples, só para fins de demonstração — deixamos isso
    # documentado também no relatório técnico).
    ruido = np.random.default_rng(42).normal(0, 3, size=len(df))
    df["pressao_sistolica"] = 100 + 0.25 * (df["frequencia_cardiaca"] - 70) + ruido
    df["pressao_diastolica"] = 70 + 0.15 * (df["frequencia_cardiaca"] - 70) + ruido

    df = df.dropna().reset_index(drop=True)
    logger.info("Registro real carregado com sucesso: %d amostras.", len(df))
    return df

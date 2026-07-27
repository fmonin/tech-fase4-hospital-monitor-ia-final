"""
Construção do conjunto de dados rotulado usado para TREINAR o classificador
de anomalias em sinais vitais.

Por que dados sintéticos para treino (e não os dados reais do VitalDB)? O
VitalDB (https://physionet.org/content/vitaldb/1.0.0/ e a API oficial em
vitaldb.net) traz sinais reais de pacientes em cirurgia — riquíssimo para
VALIDAR o modelo — mas não vem com rótulo "isto é uma anomalia" pronto para
treino supervisionado; isso exigiria anotação clínica manual, fora do escopo
desta entrega. A solução usada aqui, comum em detecção de anomalias quando
não há rótulos reais disponíveis, é treinar com dados sintéticos onde a
anomalia é INJETADA por nós (logo, o rótulo é sempre conhecido e confiável),
e depois validar o modelo treinado rodando-o sobre um caso real do VitalDB
(feito em `src/dados/carregador_vitaldb.py` + na aba "Sinais Vitais" do
frontend).

Cada exemplo do conjunto é uma JANELA de 15 minutos de sinais vitais,
resumida em estatísticas (média/desvio/mínimo/máximo) — é essa janela que o
modelo aprende a classificar como normal ou anômala.
"""

import numpy as np
import pandas as pd

DURACAO_JANELA_MIN = 15
COLUNAS_SINAIS = ["frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"]


def _gerar_janela(rng: np.random.Generator, anomala: bool) -> tuple[pd.DataFrame, str]:
    """Gera uma janela de sinais vitais, normal ou com uma anomalia injetada."""
    n = DURACAO_JANELA_MIN
    janela = pd.DataFrame({
        "frequencia_cardiaca": rng.normal(75, 3, n),
        "spo2": rng.normal(97, 0.7, n),
        "pressao_sistolica": rng.normal(115, 5, n),
        "pressao_diastolica": rng.normal(75, 3.5, n),
    })

    tipo_anomalia = "nenhuma"
    if anomala:
        tipo_anomalia = rng.choice(["dessaturacao", "taquicardia", "hipotensao", "hipertensao"])
        inicio = rng.integers(0, max(1, n - 5))
        fim = min(n, inicio + rng.integers(3, 6))

        if tipo_anomalia == "dessaturacao":
            janela.loc[inicio:fim, "spo2"] -= rng.uniform(8, 15)
        elif tipo_anomalia == "taquicardia":
            janela.loc[inicio:fim, "frequencia_cardiaca"] += rng.uniform(35, 55)
        elif tipo_anomalia == "hipotensao":
            janela.loc[inicio:fim, "pressao_sistolica"] -= rng.uniform(25, 40)
        elif tipo_anomalia == "hipertensao":
            janela.loc[inicio:fim, "pressao_sistolica"] += rng.uniform(30, 50)
            janela.loc[inicio:fim, "pressao_diastolica"] += rng.uniform(15, 25)

    return janela, tipo_anomalia


def _extrair_atributos(janela: pd.DataFrame) -> dict:
    """Resume uma janela de sinais vitais em um vetor de estatísticas simples."""
    atributos = {}
    for coluna in COLUNAS_SINAIS:
        atributos[f"{coluna}_media"] = janela[coluna].mean()
        atributos[f"{coluna}_desvio"] = janela[coluna].std()
        atributos[f"{coluna}_min"] = janela[coluna].min()
        atributos[f"{coluna}_max"] = janela[coluna].max()
    return atributos


def construir_conjunto_dados(n_por_classe: int = 500, semente: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    Gera um conjunto balanceado: metade janelas normais, metade com alguma
    anomalia injetada. Devolve (X, y) prontos para treino supervisionado.
    """
    rng = np.random.default_rng(semente)
    linhas, rotulos = [], []

    for _ in range(n_por_classe):
        janela, _ = _gerar_janela(rng, anomala=False)
        linhas.append(_extrair_atributos(janela))
        rotulos.append(0)

    for _ in range(n_por_classe):
        janela, _ = _gerar_janela(rng, anomala=True)
        linhas.append(_extrair_atributos(janela))
        rotulos.append(1)

    X = pd.DataFrame(linhas)
    y = pd.Series(rotulos, name="anomalo")

    # Evita um cenário artificialmente perfeito em dados sintéticos.
    # Pequeno ruído de rótulo aproxima o problema de ambiguidades reais.
    proporcao_ruido_rotulo = 0.04
    n_ruido = max(1, int(len(y) * proporcao_ruido_rotulo))
    indices_ruido = rng.choice(len(y), size=n_ruido, replace=False)
    y.iloc[indices_ruido] = 1 - y.iloc[indices_ruido]

    return X, y

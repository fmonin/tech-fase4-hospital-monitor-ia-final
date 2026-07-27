import pandas as pd

from src.anomalies.vitals_anomaly import DetectorAnomaliasVitais
from src.dados.dados_sinteticos import gerar_sinais_vitais_sinteticos


def test_detecta_anomalias_propositais_nos_dados_sinteticos():
    sinais = gerar_sinais_vitais_sinteticos(horas=6)
    anomalias = DetectorAnomaliasVitais().detectar(sinais)

    # Os dados sintéticos têm duas anomalias propositais (dessaturação e
    # taquicardia) — o detector precisa achar pelo menos essas.
    assert len(anomalias) > 0
    metricas_encontradas = {a["metrica"] for a in anomalias}
    assert "spo2" in metricas_encontradas or "frequencia_cardiaca" in metricas_encontradas


def test_sinais_normais_nao_geram_muitas_anomalias():
    sinais_normais = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1min"),
        "frequencia_cardiaca": [75] * 100,
        "spo2": [97] * 100,
        "pressao_sistolica": [115] * 100,
        "pressao_diastolica": [75] * 100,
    })
    anomalias = DetectorAnomaliasVitais().detectar(sinais_normais)
    assert len(anomalias) == 0


def test_levanta_erro_se_nenhuma_coluna_reconhecida():
    df_invalido = pd.DataFrame({"timestamp": [1, 2, 3], "coluna_qualquer": [1, 2, 3]})
    try:
        DetectorAnomaliasVitais().detectar(df_invalido)
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass

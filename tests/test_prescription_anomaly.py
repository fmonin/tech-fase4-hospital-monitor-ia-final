from src.anomalies.prescription_anomaly import DetectorAnomaliaPrescricao
from src.dados.dados_sinteticos import gerar_historico_prescricoes_sintetico


def test_detecta_variacao_de_dose_proposital():
    historico = gerar_historico_prescricoes_sintetico()
    anomalias = DetectorAnomaliaPrescricao().detectar(historico)

    assert len(anomalias) > 0
    assert any(a["tipo"] == "variacao_dose" for a in anomalias)


def test_prescricoes_estaveis_nao_geram_anomalia():
    import pandas as pd
    historico_estavel = pd.DataFrame([
        {"data": "2026-01-01", "medicamento": "Paracetamol", "dose_mg": 500, "frequencia_horas": 8},
        {"data": "2026-01-02", "medicamento": "Paracetamol", "dose_mg": 500, "frequencia_horas": 8},
        {"data": "2026-01-03", "medicamento": "Paracetamol", "dose_mg": 500, "frequencia_horas": 8},
    ])
    historico_estavel["data"] = pd.to_datetime(historico_estavel["data"])

    anomalias = DetectorAnomaliaPrescricao().detectar(historico_estavel)
    assert len(anomalias) == 0

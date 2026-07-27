import numpy as np

from src.anomalies.movement_anomaly import DetectorAnomaliaMovimento
from src.dados.dados_sinteticos import gerar_sequencia_pontos_sintetica


def test_detecta_evento_brusco_proposital():
    sequencia_pontos = gerar_sequencia_pontos_sintetica(n_frames=150)
    anomalias = DetectorAnomaliaMovimento().detectar(sequencia_pontos)
    assert len(anomalias) > 0
    assert all("motivos" in evento for evento in anomalias)


def test_sequencia_curta_nao_quebra_o_detector():
    sequencia_pontos = np.zeros((3, 33, 3))
    anomalias = DetectorAnomaliaMovimento().detectar(sequencia_pontos)
    assert anomalias == []

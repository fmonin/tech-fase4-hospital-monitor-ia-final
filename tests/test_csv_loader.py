from src.anomalies.vitals_anomaly import DetectorAnomaliasVitais
from src.dados.carregador_csv import carregar_sinais_vitais_de_csv

CAMINHO_AMOSTRA = "amostras/sinais_vitais_exemplo.csv"


def test_carrega_csv_de_amostra_e_mapeia_colunas():
    df = carregar_sinais_vitais_de_csv(CAMINHO_AMOSTRA)

    assert list(df.columns) == ["timestamp", "frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"]
    assert len(df) == 6


def test_deteccao_encontra_a_anomalia_da_amostra():
    df = carregar_sinais_vitais_de_csv(CAMINHO_AMOSTRA)
    anomalias = DetectorAnomaliasVitais().detectar(df)

    # A amostra tem uma linha propositalmente fora do padrão (145 bpm, SpO2 89%).
    assert len(anomalias) > 0
    metricas_encontradas = {a["metrica"] for a in anomalias}
    assert "frequencia_cardiaca" in metricas_encontradas
    assert "spo2" in metricas_encontradas


def test_csv_sem_colunas_obrigatorias_levanta_erro(tmp_path):
    caminho = tmp_path / "invalido.csv"
    caminho.write_text("coluna_qualquer\n1\n2\n")

    try:
        carregar_sinais_vitais_de_csv(str(caminho))
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass

from src.anomalies.alert_manager import GerenciadorDeAlertas
from src.fusion.multimodal_fusion import calcular_risco_paciente


def test_sem_anomalias_da_risco_baixo_e_sem_alertas():
    gerenciador = GerenciadorDeAlertas()
    resultado = calcular_risco_paciente("paciente-teste", gerenciador)

    assert resultado["nivel_risco"] == "baixo"
    assert resultado["pontuacao_risco"] == 0
    assert gerenciador.resumo_por_severidade() == {"baixa": 0, "media": 0, "alta": 0}


def test_anomalia_vital_grave_eleva_risco_e_gera_alerta():
    gerenciador = GerenciadorDeAlertas()
    anomalias_vitais = [{
        "timestamp": "2026-01-01T10:00:00", "metrica": "spo2", "valor": 85,
        "origem_deteccao": "limite_clinico", "severidade": "alta",
    }]

    resultado = calcular_risco_paciente("paciente-teste", gerenciador, anomalias_vitais=anomalias_vitais)

    assert resultado["pontuacao_risco"] > 0
    assert len(gerenciador.alertas_do_paciente("paciente-teste")) == 1
    assert gerenciador.alertas_do_paciente("paciente-teste")[0].severidade == "alta"


def test_multiplas_fontes_de_risco_se_somam():
    gerenciador = GerenciadorDeAlertas()
    anomalias_vitais = [{
        "timestamp": "2026-01-01T10:00:00", "metrica": "spo2", "valor": 85,
        "origem_deteccao": "limite_clinico", "severidade": "alta",
    }]
    anomalias_prescricao = [{
        "data": "2026-01-01", "medicamento": "Dipirona", "tipo": "variacao_dose", "severidade": "alta",
    }]

    resultado = calcular_risco_paciente(
        "paciente-teste", gerenciador,
        anomalias_vitais=anomalias_vitais,
        anomalias_prescricao=anomalias_prescricao,
    )

    assert resultado["nivel_risco"] in ("medio", "alto")
    assert len(gerenciador.alertas_do_paciente("paciente-teste")) == 2


def test_evento_sonoro_grave_do_audioset_eleva_risco():
    gerenciador = GerenciadorDeAlertas()
    resultado_audio = {
        "eventos_sonoros_audioset": [
            {"classe": "Vomiting", "pontuacao": 0.72, "relevante_clinicamente": True},
            {"classe": "Speech", "pontuacao": 0.55, "relevante_clinicamente": True},
        ],
    }

    resultado = calcular_risco_paciente("paciente-teste", gerenciador, resultado_audio=resultado_audio)

    assert resultado["pontuacao_risco"] > 0
    alertas = gerenciador.alertas_do_paciente("paciente-teste")
    assert any("YAMNet" in a.mensagem for a in alertas)


def test_evento_sonoro_comum_do_audioset_nao_eleva_risco():
    gerenciador = GerenciadorDeAlertas()
    resultado_audio = {
        "eventos_sonoros_audioset": [
            {"classe": "Speech", "pontuacao": 0.8, "relevante_clinicamente": True},
            {"classe": "Breathing", "pontuacao": 0.4, "relevante_clinicamente": True},
        ],
    }

    resultado = calcular_risco_paciente("paciente-teste", gerenciador, resultado_audio=resultado_audio)

    assert resultado["pontuacao_risco"] == 0


def test_sentimento_misto_do_audio_aparece_nos_motivos_sem_elevar_risco():
    gerenciador = GerenciadorDeAlertas()
    resultado_audio = {"analise_texto": {"sentimento": "mixed"}}

    resultado = calcular_risco_paciente("paciente-teste", gerenciador, resultado_audio=resultado_audio)

    assert resultado["pontuacao_risco"] == 0
    assert "Sentimento geral identificado na transcrição: misto" in resultado["motivos"]
    assert gerenciador.resumo_por_severidade() == {"baixa": 0, "media": 0, "alta": 0}

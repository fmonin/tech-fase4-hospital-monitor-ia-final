"""
Geração de dados sintéticos, usados como reserva quando não é possível
baixar dados reais (sem internet, ou sem credenciais Azure configuradas).

Cada função aqui gera dados com o MESMO formato que o pipeline real
produziria, incluindo algumas anomalias propositais — assim dá pra testar
os detectores de ponta a ponta mesmo em ambiente totalmente offline.
"""

import numpy as np
import pandas as pd


def gerar_sinais_vitais_sinteticos(horas: int = 24, semente: int = 7) -> pd.DataFrame:
    """
    Simula sinais vitais de um paciente internado, uma leitura por minuto,
    com duas anomalias propositais no meio da série (uma queda de SpO2 e
    um pico de frequência cardíaca) para validar o detector.
    """
    rng = np.random.default_rng(semente)
    n_amostras = horas * 60

    timestamps = pd.date_range("2026-01-01 00:00", periods=n_amostras, freq="1min")
    # Desvios-padrão deliberadamente pequenos: sinais vitais de um paciente
    # estável variam pouco minuto a minuto. Ruído maior que isso faria o
    # detector estatístico soar em cima de flutuação normal, exatamente o
    # tipo de alarme falso que sistemas de monitoramento tentam evitar.
    frequencia_cardiaca = rng.normal(75, 3, n_amostras)
    spo2 = rng.normal(97, 0.7, n_amostras)
    pressao_sistolica = rng.normal(115, 5, n_amostras)
    pressao_diastolica = rng.normal(75, 3.5, n_amostras)

    # Anomalia 1: episódio de dessaturação (SpO2 cai bruscamente)
    inicio = n_amostras // 3
    spo2[inicio:inicio + 10] -= rng.uniform(8, 14, 10)

    # Anomalia 2: taquicardia associada a pico de pressão
    inicio2 = 2 * n_amostras // 3
    frequencia_cardiaca[inicio2:inicio2 + 15] += rng.uniform(35, 50, 15)
    pressao_sistolica[inicio2:inicio2 + 15] += rng.uniform(20, 35, 15)

    return pd.DataFrame({
        "timestamp": timestamps,
        "frequencia_cardiaca": frequencia_cardiaca.clip(35, 200),
        "spo2": spo2.clip(70, 100),
        "pressao_sistolica": pressao_sistolica.clip(70, 220),
        "pressao_diastolica": pressao_diastolica.clip(40, 140),
    })


def gerar_historico_prescricoes_sintetico() -> pd.DataFrame:
    """
    Simula a evolução de prescrições médicas de um paciente ao longo da
    internação, incluindo uma alteração de dose fora do padrão esperado.
    """
    registros = [
        {"data": "2026-01-01", "medicamento": "Dipirona", "dose_mg": 500, "frequencia_horas": 6},
        {"data": "2026-01-02", "medicamento": "Dipirona", "dose_mg": 500, "frequencia_horas": 6},
        {"data": "2026-01-03", "medicamento": "Dipirona", "dose_mg": 500, "frequencia_horas": 6},
        {"data": "2026-01-04", "medicamento": "Dipirona", "dose_mg": 2000, "frequencia_horas": 4},  # anômalo
        {"data": "2026-01-05", "medicamento": "Enoxaparina", "dose_mg": 40, "frequencia_horas": 24},
        {"data": "2026-01-06", "medicamento": "Enoxaparina", "dose_mg": 40, "frequencia_horas": 24},
    ]
    df = pd.DataFrame(registros)
    df["data"] = pd.to_datetime(df["data"])
    return df


def gerar_sequencia_pontos_sintetica(n_frames: int = 150, semente: int = 3) -> np.ndarray:
    """
    Simula os 33 pontos-chave do corpo (formato MediaPipe) durante uma
    sessão de fisioterapia: postura parada com pequeno balanço natural, e um
    evento brusco (ex: perda de equilíbrio) no meio da sequência, onde o
    ombro direito cai e o tronco se inclina — o tipo de padrão que
    `src/video/pose_features.py` e o detector de anomalias de movimento
    (src/anomalies/movement_anomaly.py) devem identificar.

    Retorna um array (n_frames, 33, 3): x, y normalizados + confiança.
    """
    from src.video.pose_features import pontos_base_neutros

    rng = np.random.default_rng(semente)
    base = pontos_base_neutros()
    t = np.linspace(0, 4 * np.pi, n_frames)

    sequencia = np.repeat(base[np.newaxis, :, :], n_frames, axis=0)
    balanco = 0.01 * np.sin(t)  # leve balanço natural do corpo, de pé
    sequencia[:, :, 0] += balanco[:, np.newaxis]
    sequencia[:, :, :2] += rng.normal(0, 0.004, (n_frames, 33, 2))

    # Evento brusco: ombro direito cai e o tronco se inclina por alguns frames.
    inicio = n_frames // 2
    duracao = 6
    queda = np.linspace(0, 0.18, duracao)
    sequencia[inicio:inicio + duracao, 12, 1] += queda        # ombro direito (índice 12)
    sequencia[inicio:inicio + duracao, 24, 0] += queda * 0.5  # quadril direito acompanha

    return sequencia


def gerar_audio_sintetico(caminho_saida: str, duracao_segundos: float = 4.0, taxa_amostragem: int = 16000) -> str:
    """
    Gera um arquivo .wav sintético simples (tom + ruído) só para permitir
    testar a extração de características acústicas sem depender de um
    áudio real de consulta. Não substitui um teste com voz real.
    """
    import soundfile as sf

    rng = np.random.default_rng(1)
    t = np.linspace(0, duracao_segundos, int(taxa_amostragem * duracao_segundos))
    tom = 0.2 * np.sin(2 * np.pi * 120 * t)  # tom grave, na faixa da voz humana
    ruido = rng.normal(0, 0.02, t.shape)
    onda = (tom + ruido).astype(np.float32)

    sf.write(caminho_saida, onda, taxa_amostragem)
    return caminho_saida

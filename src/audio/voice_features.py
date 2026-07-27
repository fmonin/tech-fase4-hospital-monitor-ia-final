"""
Extração de características acústicas da voz, usadas como indício (não como
diagnóstico) de fadiga ou disartria — conforme pedido no desafio.

Importante deixar claro: isto é um MODELO HEURÍSTICO SIMPLES, pensado para
demonstrar o conceito de análise multimodal. Detecção clínica real de
disartria/fadiga por voz é um problema de pesquisa em si, normalmente
resolvido com modelos treinados em bases de fala patológica (ex: bases de
Parkinson/ELA), não com regras fixas como as daqui.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extrair_indicadores_de_fala(caminho_audio: str) -> dict:
    """
    Calcula um pequeno conjunto de métricas acústicas e as traduz em
    indicadores de risco (0 a 1) de fadiga e de alteração de fala.
    """
    onda, taxa_amostragem = librosa.load(caminho_audio, sr=None)

    duracao_segundos = librosa.get_duration(y=onda, sr=taxa_amostragem)

    # Frequência fundamental (pitch) — variação alta demais ou baixa demais
    # ao longo da fala pode indicar cansaço ou alteração motora da fala.
    f0, _, _ = librosa.pyin(onda, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_valido = f0[~np.isnan(f0)]
    variabilidade_pitch = float(np.std(f0_valido)) if len(f0_valido) > 0 else 0.0

    # Proporção de silêncio: pausas longas/frequentes podem indicar fadiga
    # ou dificuldade respiratória durante a fala.
    intervalos_com_voz = librosa.effects.split(onda, top_db=30)
    tempo_com_voz = sum((fim - inicio) for inicio, fim in intervalos_com_voz) / taxa_amostragem
    proporcao_silencio = 1 - (tempo_com_voz / duracao_segundos if duracao_segundos > 0 else 0)

    # Energia média (RMS) — voz muito "fraca" também é um indício de cansaço.
    energia_media = float(np.mean(librosa.feature.rms(y=onda)))

    indicador_fadiga = _pontuar_fadiga(proporcao_silencio, energia_media)
    indicador_alteracao_fala = _pontuar_alteracao_fala(variabilidade_pitch)

    logger.info(
        "Indicadores de voz: fadiga=%.2f, alteracao_fala=%.2f",
        indicador_fadiga, indicador_alteracao_fala,
    )

    return {
        "duracao_segundos": round(duracao_segundos, 2),
        "proporcao_silencio": round(proporcao_silencio, 3),
        "energia_media": round(energia_media, 4),
        "variabilidade_pitch_hz": round(variabilidade_pitch, 2),
        "indicador_fadiga": round(indicador_fadiga, 2),
        "indicador_alteracao_fala": round(indicador_alteracao_fala, 2),
    }


def _pontuar_fadiga(proporcao_silencio: float, energia_media: float) -> float:
    """Quanto mais pausas e menos energia na voz, maior o indicador (0 a 1)."""
    pontuacao = 0.6 * min(proporcao_silencio / 0.5, 1.0) + 0.4 * min(max(0.02 - energia_media, 0) / 0.02, 1.0)
    return float(np.clip(pontuacao, 0, 1))


def _pontuar_alteracao_fala(variabilidade_pitch: float) -> float:
    """Pitch pouco variável (fala monotônica) pode indicar disartria."""
    if variabilidade_pitch == 0:
        return 0.0
    # Fala saudável costuma ter variabilidade de pitch moderada (~20-40 Hz).
    # Valores muito abaixo disso pontuam mais alto aqui.
    pontuacao = max(0.0, 1 - (variabilidade_pitch / 20))
    return float(np.clip(pontuacao, 0, 1))

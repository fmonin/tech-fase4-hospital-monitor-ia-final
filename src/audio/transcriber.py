"""
Transcrição de áudio de consultas médicas usando o Azure Speech to Text.

Este módulo assume que o usuário tem uma chave Azure válida (configurada em
.env, ver src/config.py). Se as credenciais não estiverem presentes, a função
levanta um erro claro em vez de falhar silenciosamente — assim fica óbvio,
já no log, que falta configurar o Azure antes de rodar em produção.
"""

import logging
import math
import tempfile
from threading import Event
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src import config

logger = logging.getLogger(__name__)

TAXA_AMOSTRAGEM_AZURE = 16000


def _preparar_audio_para_speech(caminho_audio: str) -> Path:
    """Converte para WAV PCM mono/16 kHz, formato estável para o Azure Speech."""
    dados, taxa_amostragem = sf.read(caminho_audio, always_2d=True)
    audio_mono = np.mean(dados, axis=1)

    if taxa_amostragem != TAXA_AMOSTRAGEM_AZURE:
        divisor = math.gcd(taxa_amostragem, TAXA_AMOSTRAGEM_AZURE)
        audio_mono = resample_poly(
            audio_mono,
            TAXA_AMOSTRAGEM_AZURE // divisor,
            taxa_amostragem // divisor,
        )

    with tempfile.NamedTemporaryFile(prefix="azure_speech_", suffix=".wav", delete=False) as arquivo:
        caminho_normalizado = Path(arquivo.name)
    sf.write(caminho_normalizado, audio_mono, TAXA_AMOSTRAGEM_AZURE, subtype="PCM_16")
    return caminho_normalizado


def transcrever_audio(caminho_audio: str, idioma: str = "pt-BR", tempo_limite_segundos: int = 60) -> str:
    """
    Envia o arquivo de áudio para o Azure Speech to Text e devolve a
    transcrição completa (concatenando todos os trechos reconhecidos).
    """
    if not config.azure_speech_configurado():
        raise RuntimeError(
            "Azure Speech não configurado. Defina AZURE_SPEECH_KEY e "
            "AZURE_SPEECH_REGION no arquivo .env antes de transcrever áudio."
        )

    import azure.cognitiveservices.speech as speechsdk

    config_speech = speechsdk.SpeechConfig(subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION)
    config_speech.speech_recognition_language = idioma
    caminho_normalizado = _preparar_audio_para_speech(caminho_audio)
    config_audio = speechsdk.audio.AudioConfig(filename=str(caminho_normalizado))

    reconhecedor = speechsdk.SpeechRecognizer(speech_config=config_speech, audio_config=config_audio)

    trechos_reconhecidos = []
    transcricao_concluida = Event()
    erro_cancelamento = {"mensagem": None}

    def _ao_reconhecer(evento):
        if evento.result.text:
            trechos_reconhecidos.append(evento.result.text)

    def _ao_finalizar(_evento):
        transcricao_concluida.set()

    def _ao_cancelar(evento):
        detalhes = evento.result.cancellation_details
        if str(detalhes.reason) != "CancellationReason.EndOfStream":
            erro_cancelamento["mensagem"] = detalhes.error_details or str(detalhes.reason)
        transcricao_concluida.set()

    reconhecedor.recognized.connect(_ao_reconhecer)
    reconhecedor.session_stopped.connect(_ao_finalizar)
    reconhecedor.canceled.connect(_ao_cancelar)

    try:
        reconhecedor.start_continuous_recognition()
        if not transcricao_concluida.wait(tempo_limite_segundos):
            raise RuntimeError(
                f"Azure Speech não concluiu a transcrição em {tempo_limite_segundos} segundos. "
                "Verifique a conexão, a região e a chave configuradas."
            )
    finally:
        reconhecedor.stop_continuous_recognition()
        try:
            caminho_normalizado.unlink(missing_ok=True)
        except PermissionError:
            logger.warning(
                "O arquivo temporário do Azure Speech ainda está em uso e será removido depois: %s",
                caminho_normalizado,
            )

    if erro_cancelamento["mensagem"]:
        raise RuntimeError(f"Azure Speech cancelou a transcrição: {erro_cancelamento['mensagem']}")

    transcricao = " ".join(trechos_reconhecidos).strip()
    logger.info("Transcrição concluída (%d caracteres).", len(transcricao))
    return transcricao

"""
Transcrição de áudio de consultas médicas usando o Azure Speech to Text.

Este módulo assume que o usuário tem uma chave Azure válida (configurada em
.env, ver src/config.py). Se as credenciais não estiverem presentes, a função
levanta um erro claro em vez de falhar silenciosamente — assim fica óbvio,
já no log, que falta configurar o Azure antes de rodar em produção.
"""

import logging

from src import config

logger = logging.getLogger(__name__)


def transcrever_audio(caminho_audio: str, idioma: str = "pt-BR") -> str:
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
    config_audio = speechsdk.audio.AudioConfig(filename=caminho_audio)

    reconhecedor = speechsdk.SpeechRecognizer(speech_config=config_speech, audio_config=config_audio)

    trechos_reconhecidos = []
    transcricao_concluida = {"fim": False}

    def _ao_reconhecer(evento):
        if evento.result.text:
            trechos_reconhecidos.append(evento.result.text)

    def _ao_finalizar(_evento):
        transcricao_concluida["fim"] = True

    reconhecedor.recognized.connect(_ao_reconhecer)
    reconhecedor.session_stopped.connect(_ao_finalizar)
    reconhecedor.canceled.connect(_ao_finalizar)

    reconhecedor.start_continuous_recognition()
    while not transcricao_concluida["fim"]:
        import time
        time.sleep(0.5)
    reconhecedor.stop_continuous_recognition()

    transcricao = " ".join(trechos_reconhecidos).strip()
    logger.info("Transcrição concluída (%d caracteres).", len(transcricao))
    return transcricao

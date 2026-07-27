"""
Orquestra a análise de um áudio de consulta médica: transcrição (Azure
Speech to Text), análise textual (Azure Text Analytics), extração de
indicadores acústicos (fadiga/disartria) e classificação de eventos
sonoros com YAMNet (modelo treinado no AudioSet).

Cada etapa "externa" (Azure, YAMNet) fica isolada e protegida por
try/except: se o Azure não estiver configurado, ou se o TensorFlow/YAMNet
não estiver instalado ou não conseguir baixar o modelo (precisa de
internet na primeira vez), o pipeline segue com o que já tem — nunca
derruba a análise inteira do paciente por causa de uma dependência
opcional faltando. Os avisos ficam registrados em `resultado["avisos"]`.
"""

import logging
from collections.abc import Callable

from src.audio.risco_fala import classificar_risco_fala
from src.audio.text_analyzer import analisar_texto
from src.audio.transcriber import transcrever_audio
from src.audio.voice_features import extrair_indicadores_de_fala

logger = logging.getLogger(__name__)


def processar_audio_consulta(caminho_audio: str, status_callback: Callable[[str], None] | None = None) -> dict:
    """Executa o pipeline completo de áudio e devolve um dicionário-relatório."""
    logger.info("Processando áudio de consulta: %s", caminho_audio)

    resultado = {"audio": caminho_audio, "avisos": []}

    resultado["indicadores_acusticos"] = extrair_indicadores_de_fala(caminho_audio)

    # Classificação pelo modelo treinado (RandomForest); se ainda não foi
    # treinado, fica None e o resto do sistema usa só o indicador heurístico.
    resultado["classificacao_modelo_treinado"] = classificar_risco_fala(resultado["indicadores_acusticos"])

    try:
        from src.audio.audioset_eventos import classificar_eventos_sonoros
        resultado["eventos_sonoros_audioset"] = classificar_eventos_sonoros(caminho_audio)
    except Exception as erro:
        logger.warning("Classificação de eventos sonoros (YAMNet/AudioSet) pulada: %s", erro)
        resultado["eventos_sonoros_audioset"] = None
        resultado["avisos"].append(f"YAMNet/AudioSet indisponível: {erro}")

    try:
        if status_callback:
            status_callback("Extraindo texto com AZure 'Speech to Text'.")
        transcricao = transcrever_audio(caminho_audio)
        resultado["transcricao"] = transcricao

        if status_callback:
            status_callback("Analisando Texto com AZure 'Text Analytics'.")
        resultado["analise_texto"] = analisar_texto(transcricao)

        if status_callback:
            status_callback("Análise concluída com AZure 'Text Analytics'.")
    except RuntimeError as erro:
        # Azure não configurado: seguimos com o que já temos (características
        # acústicas) e deixamos o aviso registrado no relatório final.
        logger.warning("Etapa de transcrição/análise textual pulada: %s", erro)
        resultado["transcricao"] = None
        resultado["analise_texto"] = None
        resultado["avisos"].append(str(erro))

    return resultado

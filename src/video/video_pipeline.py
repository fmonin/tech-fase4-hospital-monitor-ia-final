"""
Orquestra a análise de um vídeo clínico (fisioterapia ou cirurgia):
1. extrai os pontos-chave de postura, corpo inteiro (AnalisadorDePostura);
2. roda a detecção de objetos/pessoas em cena (DetectorDeObjetos);
3. passa a sequência de pontos pelo detector de anomalias de movimento;
4. opcionalmente, gera uma cópia anotada do vídeo (esqueleto + caixas +
   aviso nos frames anômalos) para facilitar a demonstração visual;
5. devolve um relatório único, pronto para entrar no relatório final
   e alimentar a fusão multimodal.
"""

import logging

from src.anomalies.movement_anomaly import DetectorAnomaliaMovimento
from src.video.object_detector import DetectorDeObjetos
from src.video.pose_analyzer import AnalisadorDePostura

logger = logging.getLogger(__name__)


def processar_video_clinico(caminho_video: str, caminho_video_anotado: str | None = None) -> dict:
    """
    Executa o pipeline completo de vídeo e devolve um dicionário-relatório.

    Se `caminho_video_anotado` for informado, também gera nesse caminho uma
    cópia do vídeo com o esqueleto do MediaPipe, as caixas do YOLOv8 e um
    aviso em vermelho sobre os frames marcados como anômalos — útil para
    exibir na aba "Vídeo" do frontend ou no vídeo de demonstração.
    """
    logger.info("Processando vídeo clínico: %s", caminho_video)

    analisador_postura = AnalisadorDePostura()
    sequencia_pontos = analisador_postura.extrair_pontos_completos(caminho_video)

    detector_objetos = DetectorDeObjetos()
    eventos_objetos = detector_objetos.detectar_em_video(caminho_video)
    resumo_objetos = detector_objetos.resumir_deteccoes(eventos_objetos)
    eventos_areas_criticas = detector_objetos.detectar_eventos_em_areas_criticas(eventos_objetos)

    detector_movimento = DetectorAnomaliaMovimento()
    anomalias_movimento = detector_movimento.detectar(sequencia_pontos)

    desvio_detectado = len(anomalias_movimento) > 0 or len(eventos_areas_criticas) > 0

    resultado = {
        "video": caminho_video,
        "backend_postura": analisador_postura.backend_em_uso,
        "total_frames_analisados": len(sequencia_pontos),
        "objetos_detectados": resumo_objetos,
        "eventos_areas_criticas": eventos_areas_criticas,
        "anomalias_movimento": anomalias_movimento,
        "possivel_desvio_no_procedimento": desvio_detectado,
        "video_anotado": None,
    }

    if caminho_video_anotado:
        try:
            from src.video.video_anotado import gerar_video_anotado
            frames_anomalos = _extrair_frames_anomalos(anomalias_movimento)
            resultado["video_anotado"] = gerar_video_anotado(caminho_video, caminho_video_anotado, frames_anomalos)
        except Exception as erro:
            logger.warning("Não foi possível gerar o vídeo anotado: %s", erro)

    return resultado


def _extrair_frames_anomalos(anomalias_movimento: list[dict]) -> set:
    """
    Os eventos vêm com "janela": [inicio, fim] (caminho do modelo treinado)
    ou só "frame" (caminho de fallback por regra) — normalizamos os dois
    formatos num único conjunto de índices de frame a destacar no vídeo.
    """
    frames = set()
    for evento in anomalias_movimento:
        if "janela" in evento:
            inicio, fim = evento["janela"]
            frames.update(range(inicio, fim + 1))
        elif "frame" in evento:
            frames.add(evento["frame"])
    return frames

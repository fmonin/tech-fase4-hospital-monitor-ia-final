"""
Detecção de objetos e áreas críticas em vídeo, usando YOLOv8 (conforme
sugerido no enunciado do desafio).

Aqui o objetivo não é reconhecer instrumentos cirúrgicos específicos (isso
exigiria treinar um modelo com dataset médico próprio, fora do escopo desta
entrega) — usamos o modelo YOLOv8 pré-treinado (COCO) para detectar e
localizar pessoas em cena, o que já é útil para, por exemplo, contar quantas
pessoas estão na sala durante um procedimento ou confirmar presença contínua
do paciente/fisioterapeuta em quadro. A arquitetura fica pronta para trocar
o modelo por um YOLOv8 fine-tuned em objetos médicos, quando esse dataset
rotulado existir.
"""

import logging
from collections import Counter

from src import config

logger = logging.getLogger(__name__)


class DetectorDeObjetos:
    """Roda o YOLOv8 sobre um vídeo e resume o que foi detectado por frame."""

    def __init__(self, modelo: str = "yolov8n.pt", confianca_minima: float = 0.4):
        from ultralytics import YOLO

        # "yolov8n.pt" (nano) é baixado automaticamente pela biblioteca ultralytics
        # na primeira execução, direto do repositório oficial do modelo.
        self._modelo = YOLO(modelo)
        self._confianca_minima = confianca_minima

    def detectar_em_video(self, caminho_video: str, a_cada_n_frames: int = 5) -> list[dict]:
        """
        Roda a detecção a cada N frames (não em todos, para não deixar o
        processamento lento demais) e devolve uma lista de eventos:

            {"frame": int, "classe": "person", "confianca": 0.91}
        """
        import cv2

        captura = cv2.VideoCapture(caminho_video)
        largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
        altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        captura.release()

        resultados_por_frame = self._modelo.predict(
            source=caminho_video,
            conf=self._confianca_minima,
            stream=True,
            verbose=False,
            vid_stride=a_cada_n_frames,
        )

        eventos = []
        for indice_frame, resultado in enumerate(resultados_por_frame):
            for caixa in resultado.boxes:
                classe = resultado.names[int(caixa.cls[0])]
                confianca = float(caixa.conf[0])
                x1, y1, x2, y2 = [int(v) for v in caixa.xyxy[0]]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                eventos.append({
                    "frame": indice_frame * a_cada_n_frames,
                    "classe": classe,
                    "confianca": round(confianca, 3),
                    "caixa": [x1, y1, x2, y2],
                    "centro_normalizado": [round(cx / largura, 4), round(cy / altura, 4)],
                })

        logger.info("Detecção concluída: %d objetos encontrados.", len(eventos))
        return eventos

    def resumir_deteccoes(self, eventos: list[dict]) -> dict:
        """Conta quantas vezes cada classe de objeto apareceu no vídeo."""
        contagem = Counter(evento["classe"] for evento in eventos)
        return dict(contagem)

    def detectar_eventos_em_areas_criticas(
        self,
        eventos: list[dict],
        areas_criticas: list[dict] | None = None,
        classes_monitoradas: tuple[str, ...] = ("person",),
        confianca_minima: float = 0.4,
    ) -> list[dict]:
        """
        Marca eventos onde o centro da caixa detectada entra em uma área crítica.
        As áreas são retângulos normalizados (0-1) configurados em src/config.py.
        """
        areas = areas_criticas or config.AREAS_CRITICAS_VIDEO
        eventos_criticos = []

        for evento in eventos:
            if evento.get("classe") not in classes_monitoradas:
                continue
            if float(evento.get("confianca", 0.0)) < confianca_minima:
                continue

            centro = evento.get("centro_normalizado")
            if not centro or len(centro) != 2:
                continue
            cx, cy = centro

            for area in areas:
                if area["xmin"] <= cx <= area["xmax"] and area["ymin"] <= cy <= area["ymax"]:
                    eventos_criticos.append({
                        "frame": evento["frame"],
                        "classe": evento["classe"],
                        "confianca": evento["confianca"],
                        "area_critica": area["nome"],
                        "caixa": evento.get("caixa"),
                    })

        logger.info("Eventos em áreas críticas: %d", len(eventos_criticos))
        return eventos_criticos

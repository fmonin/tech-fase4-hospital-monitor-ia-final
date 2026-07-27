"""
Gera uma cópia anotada do vídeo processado: esqueleto do MediaPipe
desenhado sobre a pessoa, caixas do YOLOv8 quando há objeto/pessoa
detectado, e um aviso em vermelho nos frames onde o modelo treinado de
movimento identificou uma anomalia (ver src/anomalies/movement_anomaly.py).

Fica separado do resto do pipeline de vídeo (video_pipeline.py) de
propósito: a extração de trajetória e a detecção de anomalias já atendem
sozinhas ao que o desafio pede ("gerar relatórios automáticos indicando
desvios"); isto aqui é só a camada de apresentação — pensada para tornar a
demonstração visual mais fácil de acompanhar numa gravação de tela.
"""

import logging

logger = logging.getLogger(__name__)

COR_ALERTA = (0, 0, 255)   # vermelho, em BGR (OpenCV)
COR_CAIXA = (0, 200, 0)    # verde
INTERVALO_DETECCAO_OBJETOS = 5  # roda o YOLOv8 a cada N frames, para não pesar demais


def gerar_video_anotado(caminho_video: str, caminho_saida: str, frames_anomalos: set) -> str:
    """
    Reprocessa o vídeo desenhando o esqueleto (MediaPipe), as caixas do
    YOLOv8 e um aviso nos frames marcados como anômalos. Devolve o caminho
    do vídeo gerado (mesma pasta/nome passados em `caminho_saida`).
    """
    import cv2
    import mediapipe as mp
    from ultralytics import YOLO

    mp_pose = mp.solutions.pose
    mp_desenho = mp.solutions.drawing_utils
    detector_pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    modelo_yolo = YOLO("yolov8n.pt")

    video = cv2.VideoCapture(caminho_video)
    if not video.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = video.get(cv2.CAP_PROP_FPS) or 24
    largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escritor = cv2.VideoWriter(str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

    ultimas_caixas = []
    indice_frame = 0
    try:
        while True:
            lido, frame = video.read()
            if not lido:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado_pose = detector_pose.process(frame_rgb)
            if resultado_pose.pose_landmarks:
                mp_desenho.draw_landmarks(frame, resultado_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if indice_frame % INTERVALO_DETECCAO_OBJETOS == 0:
                resultado_yolo = modelo_yolo.predict(frame, verbose=False, conf=0.4)[0]
                ultimas_caixas = [
                    (tuple(map(int, caixa.xyxy[0])), resultado_yolo.names[int(caixa.cls[0])], float(caixa.conf[0]))
                    for caixa in resultado_yolo.boxes
                ]

            for (x1, y1, x2, y2), nome, confianca in ultimas_caixas:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COR_CAIXA, 2)
                cv2.putText(frame, f"{nome} {confianca:.0%}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CAIXA, 1)

            if indice_frame in frames_anomalos:
                cv2.putText(frame, "ALERTA: movimento fora do padrao", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_ALERTA, 2)

            escritor.write(frame)
            indice_frame += 1
    finally:
        video.release()
        escritor.release()

    logger.info("Vídeo anotado gerado: %s (%d frames).", caminho_saida, indice_frame)
    return caminho_saida

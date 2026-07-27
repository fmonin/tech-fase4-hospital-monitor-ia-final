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
import subprocess
from pathlib import Path

from src.config import RAIZ_PROJETO

logger = logging.getLogger(__name__)

COR_ALERTA = (0, 0, 255)   # vermelho, em BGR (OpenCV)
COR_CAIXA = (0, 200, 0)    # verde
INTERVALO_DETECCAO_OBJETOS = 5  # roda o YOLOv8 a cada N frames, para não pesar demais
PONTOS_POR_MOTIVO = {
    "ombros": {11, 12},
    "quadril": {23, 24},
    "tronco": {11, 12, 23, 24},
    "joelho esquerdo": {23, 25, 27},
    "joelho direito": {24, 26, 28},
    "cotovelo esquerdo": {11, 13, 15},
    "cotovelo direito": {12, 14, 16},
    "velocidade": {11, 12, 23, 24},
}
CONEXOES_ESQUELETO = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def gerar_video_anotado(
    caminho_video: str,
    caminho_saida: str,
    frames_anomalos: set,
    marcacoes_anomalias: dict[int, list[str]] | None = None,
) -> str:
    """
    Reprocessa o vídeo desenhando o esqueleto (MediaPipe), as caixas do
    YOLOv8 e um aviso nos frames marcados como anômalos. As articulações
    relacionadas aos motivos do alerta são desenhadas em vermelho.
    """
    import cv2
    import mediapipe as mp
    from ultralytics import YOLO

    usar_solutions, detector_pose = _criar_detector_pose(mp)
    modelo_yolo = YOLO("yolov8n.pt")

    video = cv2.VideoCapture(caminho_video)
    if not video.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = video.get(cv2.CAP_PROP_FPS) or 24
    largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    caminho_saida = Path(caminho_saida)
    caminho_intermediario = caminho_saida.with_name(f"{caminho_saida.stem}_intermediario.mp4")
    escritor = cv2.VideoWriter(
        str(caminho_intermediario),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )
    if not escritor.isOpened():
        raise RuntimeError("Não foi possível abrir o codificador temporário de vídeo.")

    ultimas_caixas = []
    marcacoes_anomalias = marcacoes_anomalias or {}
    indice_frame = 0
    timestamp_ms = 0
    intervalo_ms = max(1, int(round(1000 / fps)))
    try:
        while True:
            lido, frame = video.read()
            if not lido:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if usar_solutions:
                resultado_pose = detector_pose.process(frame_rgb)
                landmarks = resultado_pose.pose_landmarks.landmark if resultado_pose.pose_landmarks else None
                if landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, resultado_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    )
            else:
                timestamp_ms += intervalo_ms
                imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                resultado_pose = detector_pose.detect_for_video(imagem, timestamp_ms)
                landmarks = resultado_pose.pose_landmarks[0] if resultado_pose.pose_landmarks else None
                if landmarks:
                    _desenhar_esqueleto(frame, landmarks)

            if landmarks:
                motivos = marcacoes_anomalias.get(indice_frame, [])
                _destacar_articulacoes_anomalas(frame, landmarks, motivos)

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
                motivos = marcacoes_anomalias.get(indice_frame, [])
                _desenhar_alerta(frame, motivos)

            escritor.write(frame)
            indice_frame += 1
    finally:
        video.release()
        escritor.release()
        if not usar_solutions:
            detector_pose.close()

    try:
        _converter_para_h264(caminho_intermediario, caminho_saida)
    finally:
        caminho_intermediario.unlink(missing_ok=True)

    logger.info("Vídeo anotado gerado: %s (%d frames).", caminho_saida, indice_frame)
    return str(caminho_saida)


def _converter_para_h264(caminho_entrada: Path, caminho_saida: Path) -> None:
    """Converte o MP4 intermediário para H.264, suportado pelo player do navegador."""
    try:
        import imageio_ffmpeg
    except ImportError as erro:
        raise RuntimeError(
            "A dependência imageio-ffmpeg é necessária para gerar um vídeo reproduzível no navegador. "
            "Instale as dependências do projeto novamente."
        ) from erro

    comando = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(caminho_entrada),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(caminho_saida),
    ]
    processo = subprocess.run(comando, capture_output=True, text=True, check=False)
    if processo.returncode != 0 or not caminho_saida.exists() or caminho_saida.stat().st_size == 0:
        detalhe = processo.stderr.strip().splitlines()[-1] if processo.stderr.strip() else "erro desconhecido"
        raise RuntimeError(f"Falha ao converter vídeo anotado para H.264: {detalhe}")


def _criar_detector_pose(mp):
    if hasattr(mp, "solutions"):
        detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return True, detector

    from mediapipe.tasks import python as mp_python_tasks
    from mediapipe.tasks.python import vision as mp_vision_tasks

    caminho_modelo = RAIZ_PROJETO / "models" / "pose_landmarker_full.task"
    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Modelo PoseLandmarker não encontrado em: {caminho_modelo}")
    opcoes = mp_vision_tasks.PoseLandmarkerOptions(
        base_options=mp_python_tasks.BaseOptions(model_asset_buffer=caminho_modelo.read_bytes()),
        running_mode=mp_vision_tasks.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    return False, mp_vision_tasks.PoseLandmarker.create_from_options(opcoes)


def _desenhar_esqueleto(frame, landmarks) -> None:
    import cv2

    altura, largura = frame.shape[:2]
    for inicio, fim in CONEXOES_ESQUELETO:
        ponto_inicio, ponto_fim = landmarks[inicio], landmarks[fim]
        if min(getattr(ponto_inicio, "visibility", 1.0), getattr(ponto_fim, "visibility", 1.0)) < 0.4:
            continue
        origem = (int(ponto_inicio.x * largura), int(ponto_inicio.y * altura))
        destino = (int(ponto_fim.x * largura), int(ponto_fim.y * altura))
        cv2.line(frame, origem, destino, (0, 220, 255), 2)
    for ponto in landmarks:
        if getattr(ponto, "visibility", 1.0) >= 0.4:
            cv2.circle(frame, (int(ponto.x * largura), int(ponto.y * altura)), 3, (0, 220, 255), -1)


def _pontos_relacionados_aos_motivos(motivos: list[str]) -> set[int]:
    pontos = set()
    texto_motivos = " ".join(motivos).lower()
    for termo, indices in PONTOS_POR_MOTIVO.items():
        if termo in texto_motivos:
            pontos.update(indices)
    return pontos or {11, 12, 23, 24}


def _destacar_articulacoes_anomalas(frame, landmarks, motivos: list[str]) -> None:
    import cv2

    altura, largura = frame.shape[:2]
    for indice in _pontos_relacionados_aos_motivos(motivos):
        ponto = landmarks[indice]
        if getattr(ponto, "visibility", 1.0) < 0.4:
            continue
        x, y = int(ponto.x * largura), int(ponto.y * altura)
        cv2.circle(frame, (x, y), 9, COR_ALERTA, -1)
        cv2.circle(frame, (x, y), 13, (255, 255, 255), 2)


def _desenhar_alerta(frame, motivos: list[str]) -> None:
    import cv2

    texto = "ALERTA: MOVIMENTO FORA DO PADRÃO"
    cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 700), 100), (255, 255, 255), -1)
    cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 700), 100), COR_ALERTA, 3)
    cv2.putText(frame, texto, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.78, COR_ALERTA, 2)
    if motivos:
        resumo = " | ".join(motivos[:2])
        cv2.putText(frame, resumo[:85], (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COR_ALERTA, 1)

"""
Análise postural em vídeo.

O enunciado do desafio sugere o OpenPose para análise postural. Optamos por
usar o MediaPipe Pose no lugar dele, pelo mesmo motivo: MediaPipe extrai os
33 pontos-chave do corpo (ombros, cotovelos, quadris, joelhos etc.) a partir
de um vídeo comum, sem precisar de GPU nem de uma instalação C++/CUDA
complexa como o OpenPose exige.

Extraímos o corpo INTEIRO por frame (não só um ponto-chave) para alimentar
`src/video/pose_features.py`, que calcula atributos clínicos explicáveis
(assimetria de ombros/quadril, ângulos de joelho/cotovelo, inclinação do
tronco) — uma leitura mais próxima do que um fisioterapeuta observaria do
que só "a velocidade do pulso".
"""

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from src import config

logger = logging.getLogger(__name__)

URL_MODELO_POSE_TASK = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


class AnalisadorDePostura:
    """Extrai os 33 pontos-chave do corpo, frame a frame, ao longo de um vídeo."""

    # Mapeamento OpenPose BODY_25 -> layout de 33 pontos esperado no pipeline
    # (preenchemos os pontos principais usados nos atributos clínicos; o restante fica NaN).
    _OPENPOSE_TO_LAYOUT33 = {
        0: 0,    # nariz
        15: 2,   # olho esquerdo
        16: 5,   # olho direito
        17: 7,   # orelha esquerda
        18: 8,   # orelha direita
        5: 11,   # ombro esquerdo
        2: 12,   # ombro direito
        6: 13,   # cotovelo esquerdo
        3: 14,   # cotovelo direito
        7: 15,   # punho esquerdo
        4: 16,   # punho direito
        12: 23,  # quadril esquerdo
        9: 24,   # quadril direito
        13: 25,  # joelho esquerdo
        10: 26,  # joelho direito
        14: 27,  # tornozelo esquerdo
        11: 28,  # tornozelo direito
        19: 31,  # pe esquerdo (aprox.)
        22: 32,  # pe direito (aprox.)
    }

    def __init__(self, confianca_minima: float = 0.5, backend: str | None = None):
        self._backend = (backend or config.POSE_BACKEND or "auto").lower().strip()
        self._nome_backend_em_uso = "desconhecido"
        self._detector = None
        self._mp_pose = None
        self._op = None
        self._op_wrapper = None
        self._mp_tasks_landmarker = None
        self._usa_mp_tasks = False
        self._ultimo_timestamp_ms = -1

        if self._backend in {"auto", "openpose"} and self._tentar_inicializar_openpose():
            self._nome_backend_em_uso = "openpose"
            logger.info("Backend de postura em uso: OpenPose")
            return

        if self._backend == "openpose":
            raise RuntimeError(
                "POSE_BACKEND=openpose, mas OpenPose não está disponível. "
                "Configure pyopenpose + OPENPOSE_MODEL_FOLDER, ou use POSE_BACKEND=mediapipe/auto."
            )

        # Fallback padrão para manter execução local simples quando OpenPose não existe.
        self._inicializar_mediapipe(confianca_minima)
        self._nome_backend_em_uso = "mediapipe"
        logger.info("Backend de postura em uso: MediaPipe")

    @property
    def backend_em_uso(self) -> str:
        return self._nome_backend_em_uso

    def _inicializar_mediapipe(self, confianca_minima: float) -> None:
        import mediapipe as mp

        if hasattr(mp, "solutions"):
            self._mp_pose = mp.solutions.pose
            self._detector = self._mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=confianca_minima,
                min_tracking_confidence=confianca_minima,
            )
            self._usa_mp_tasks = False
            return

        logger.warning("mediapipe.solutions indisponível; usando fallback PoseLandmarker (tasks API).")
        from mediapipe.tasks import python as mp_python_tasks
        from mediapipe.tasks.python import vision as mp_vision_tasks

        caminho_modelo = self._obter_modelo_pose_task()
        modelo_bytes = caminho_modelo.read_bytes()
        opcoes = mp_vision_tasks.PoseLandmarkerOptions(
            # Evita problemas de encoding de caminho em bibliotecas nativas
            # (ex.: diretórios com acentos no Windows).
            base_options=mp_python_tasks.BaseOptions(model_asset_buffer=modelo_bytes),
            running_mode=mp_vision_tasks.RunningMode.VIDEO,
            min_pose_detection_confidence=confianca_minima,
            min_pose_presence_confidence=confianca_minima,
            min_tracking_confidence=confianca_minima,
            num_poses=1,
        )
        self._mp_tasks_landmarker = mp_vision_tasks.PoseLandmarker.create_from_options(opcoes)
        self._usa_mp_tasks = True

    @staticmethod
    def _obter_modelo_pose_task() -> Path:
        if config.MEDIAPIPE_POSETASK_MODEL_PATH:
            caminho = Path(config.MEDIAPIPE_POSETASK_MODEL_PATH)
            if not caminho.exists():
                raise FileNotFoundError(f"Modelo de pose task não encontrado em: {caminho}")
            return caminho

        caminho = config.RAIZ_PROJETO / "models" / "pose_landmarker_full.task"
        if caminho.exists():
            return caminho

        caminho.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando modelo PoseLandmarker para %s", caminho)
        urlretrieve(URL_MODELO_POSE_TASK, caminho)
        return caminho

    @staticmethod
    def _extrair_pontos_mp_tasks(resultado) -> np.ndarray | None:
        if not getattr(resultado, "pose_landmarks", None):
            return None
        primeira_pose = resultado.pose_landmarks[0]
        pontos = []
        for p in primeira_pose:
            conf = getattr(p, "visibility", None)
            if conf is None:
                conf = getattr(p, "presence", 1.0)
            pontos.append([float(p.x), float(p.y), float(conf)])
        return np.array(pontos, dtype=float)

    def _tentar_inicializar_openpose(self) -> bool:
        try:
            from openpose import pyopenpose as op
        except Exception as erro:
            logger.warning("OpenPose indisponível no ambiente: %s", erro)
            return False

        parametros = {}
        if config.OPENPOSE_MODEL_FOLDER:
            parametros["model_folder"] = str(Path(config.OPENPOSE_MODEL_FOLDER))

        try:
            wrapper = op.WrapperPython()
            wrapper.configure(parametros)
            wrapper.start()
        except Exception as erro:
            logger.warning("Falha ao inicializar OpenPose: %s", erro)
            return False

        self._op = op
        self._op_wrapper = wrapper
        return True

    @classmethod
    def _openpose_para_layout33(cls, keypoints_person: np.ndarray, largura: int, altura: int) -> np.ndarray:
        pontos = np.full((33, 3), np.nan, dtype=float)
        if keypoints_person is None:
            return pontos

        for idx_openpose, idx_layout in cls._OPENPOSE_TO_LAYOUT33.items():
            if idx_openpose >= len(keypoints_person):
                continue
            x, y, conf = keypoints_person[idx_openpose]
            if largura > 0 and altura > 0:
                pontos[idx_layout, 0] = float(x) / float(largura)
                pontos[idx_layout, 1] = float(y) / float(altura)
            else:
                pontos[idx_layout, 0] = float(x)
                pontos[idx_layout, 1] = float(y)
            pontos[idx_layout, 2] = float(conf)
        return pontos

    def extrair_pontos_completos(self, caminho_video: str) -> np.ndarray:
        """
        Processa o vídeo inteiro e devolve um array (n_frames, 33, 3) com
        x, y (normalizados, 0-1) e confiança (visibility) de cada um dos 33
        pontos-chave do MediaPipe, em cada frame — é essa sequência que
        alimenta `pose_features.extrair_atributos_pose()` e, por fim, o
        detector de anomalias de movimento
        (src/anomalies/movement_anomaly.py).
        """
        import cv2

        video = cv2.VideoCapture(caminho_video)
        if not video.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho_video}")

        # No backend de tasks do MediaPipe, o timestamp precisa ser estritamente
        # crescente inclusive entre vídeos processados no mesmo objeto.
        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        intervalo_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 33
        timestamp_atual_ms = self._ultimo_timestamp_ms

        sequencia = []
        try:
            while True:
                lido, frame = video.read()
                if not lido:
                    break

                if self._nome_backend_em_uso == "openpose":
                    altura, largura = frame.shape[:2]
                    datum = self._op.Datum()
                    datum.cvInputData = frame
                    self._op_wrapper.emplaceAndPop([datum])

                    # Usa a pessoa com maior confiança média quando há múltiplas.
                    if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                        keypoints = datum.poseKeypoints
                        medias = np.nanmean(keypoints[:, :, 2], axis=1)
                        melhor = int(np.nanargmax(medias))
                        pontos_frame = self._openpose_para_layout33(keypoints[melhor], largura, altura)
                    else:
                        pontos_frame = sequencia[-1].copy() if sequencia else np.full((33, 3), np.nan)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self._usa_mp_tasks:
                        import mediapipe as mp

                        timestamp_atual_ms += intervalo_ms
                        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        resultado = self._mp_tasks_landmarker.detect_for_video(imagem, timestamp_atual_ms)
                        pontos_frame = self._extrair_pontos_mp_tasks(resultado)
                    else:
                        resultado = self._detector.process(frame_rgb)
                        if resultado.pose_landmarks:
                            pontos_frame = np.array(
                                [[p.x, p.y, p.visibility] for p in resultado.pose_landmarks.landmark]
                            )
                        else:
                            pontos_frame = None

                    if pontos_frame is None:
                        # Sem detecção nesse frame (ex: paciente fora de quadro):
                        # repetimos o último frame conhecido em vez de descartar,
                        # para não quebrar a continuidade da série temporal.
                        pontos_frame = sequencia[-1].copy() if sequencia else np.full((33, 3), np.nan)
                sequencia.append(pontos_frame)
                if self._usa_mp_tasks:
                    self._ultimo_timestamp_ms = timestamp_atual_ms
        finally:
            video.release()

        logger.info("Postura extraída: %d frames processados.", len(sequencia))
        return np.array(sequencia)

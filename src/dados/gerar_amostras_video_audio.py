"""
Gera os arquivos REAIS de vídeo e áudio usados nos testes e no treino com
dados reais (pasta `amostras/`). Nada aqui é baixado da internet: os
arquivos são sintetizados por código e gravados em disco, para que o
projeto venha com exemplos de verdade, prontos para abrir/tocar/inspecionar
— em vez de depender do usuário conseguir seus próprios vídeos/áudios antes
de conseguir rodar qualquer coisa.

Rodar:  python -m src.dados.gerar_amostras_video_audio

Vídeo: desenhamos uma pessoa simples (cabeça, tronco, braços, pernas) se
movendo em frente à câmera. Não é uma gravação de uma pessoa de verdade,
mas tem forma humana o suficiente para o MediaPipe Pose reconhecer e
extrair pontos-chave normalmente — testado e confirmado neste projeto.

Áudio: geramos um tom simples com ruído, com parâmetros diferentes para a
classe "normal" (tom estável) e "alterado" (tom mais instável, mais pausas)
— serve para exercitar o pipeline de ponta a ponta, não para treinar um
classificador de voz clinicamente validado.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import PASTA_AMOSTRAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gerar_amostras")

COR_PELE = (150, 190, 230)
LARGURA, ALTURA = 480, 640


def _desenhar_pessoa(frame: np.ndarray, centro_x: int, inclinacao: float, braco_erguido: float) -> None:
    """Desenha uma pessoa simples (cabeça, tronco, braços, pernas) num frame."""
    deslocamento = int(inclinacao * 40)
    cx_cabeca = centro_x + deslocamento
    cx_tronco = centro_x + deslocamento // 2

    cv2.circle(frame, (cx_cabeca, 90), 35, COR_PELE, -1)
    cv2.ellipse(frame, (cx_tronco, 220), (55, 110), int(inclinacao * 10), 0, 360, COR_PELE, -1)

    altura_braco_esquerdo = 260 - int(braco_erguido * 100)
    altura_braco_direito = 260 - int(braco_erguido * 40)
    cv2.line(frame, (cx_tronco - 50, 150), (cx_tronco - 90, altura_braco_esquerdo), COR_PELE, 20)
    cv2.line(frame, (cx_tronco + 50, 150), (cx_tronco + 90, altura_braco_direito), COR_PELE, 20)

    cv2.line(frame, (cx_tronco - 25, 320), (cx_tronco - 35 + deslocamento // 2, 480), COR_PELE, 25)
    cv2.line(frame, (cx_tronco + 25, 320), (cx_tronco + 35 + deslocamento // 2, 480), COR_PELE, 25)


def gerar_video(caminho_saida: Path, n_frames: int = 90, com_anomalia: bool = False, semente: int = 1) -> Path:
    """
    Gera um vídeo .mp4 de uma pessoa com balanço natural (normal) ou com um
    evento brusco de inclinação/queda de braço no meio (com_anomalia=True).
    """
    rng = np.random.default_rng(semente)
    escritor = cv2.VideoWriter(str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), 24, (LARGURA, ALTURA))

    inicio_evento = n_frames // 2
    duracao_evento = 15

    for indice in range(n_frames):
        frame = np.full((ALTURA, LARGURA, 3), 235, dtype=np.uint8)
        balanco = 0.05 * np.sin(indice / 6) + rng.normal(0, 0.01)
        braco_erguido = 0.0

        if com_anomalia and inicio_evento <= indice < inicio_evento + duracao_evento:
            progresso = (indice - inicio_evento) / duracao_evento
            balanco += 0.6 * progresso
            braco_erguido = 0.5 * progresso

        _desenhar_pessoa(frame, LARGURA // 2, balanco, braco_erguido)
        escritor.write(frame)

    escritor.release()
    logger.info("Vídeo gerado: %s (%d frames, anomalia=%s)", caminho_saida, n_frames, com_anomalia)
    return caminho_saida


def gerar_audio(caminho_saida: Path, classe: str, semente: int = 1) -> Path:
    """
    Gera um áudio .wav simples. Classe "normal": tom estável, pouco ruído.
    Classe "alterado": tom mais instável, mais silêncio, energia menor —
    simula indícios de fadiga/alteração de fala.
    """
    import soundfile as sf

    rng = np.random.default_rng(semente)
    duracao, taxa_amostragem = 4.0, 16000
    t = np.linspace(0, duracao, int(taxa_amostragem * duracao))

    if classe == "normal":
        frequencia = 120 + 5 * np.sin(2 * np.pi * 0.5 * t)
        onda = 0.25 * np.sin(2 * np.pi * frequencia * t) + rng.normal(0, 0.01, t.shape)
    else:
        frequencia = 100 + 25 * np.sin(2 * np.pi * 1.5 * t)
        onda = 0.12 * np.sin(2 * np.pi * frequencia * t) + rng.normal(0, 0.03, t.shape)
        onda[: len(t) // 4] = rng.normal(0, 0.005, len(t) // 4)  # trecho de silêncio, simulando pausa na fala

    sf.write(caminho_saida, onda.astype(np.float32), taxa_amostragem)
    logger.info("Áudio gerado: %s (classe=%s)", caminho_saida, classe)
    return caminho_saida


def gerar_todas_as_amostras() -> None:
    pasta_videos = PASTA_AMOSTRAS / "videos_normais"
    pasta_videos.mkdir(parents=True, exist_ok=True)
    for indice in range(1, 4):
        gerar_video(pasta_videos / f"normal_{indice}.mp4", semente=indice)
    gerar_video(PASTA_AMOSTRAS / "video_exemplo_com_anomalia.mp4", com_anomalia=True, semente=10)

    pasta_audios = PASTA_AMOSTRAS / "audio_por_classe"
    for classe in ("normal", "alterado"):
        pasta_classe = pasta_audios / classe
        pasta_classe.mkdir(parents=True, exist_ok=True)
        for indice in range(1, 6):
            gerar_audio(pasta_classe / f"{classe}_{indice}.wav", classe=classe, semente=indice)

    print("Amostras geradas em:")
    print(f"  {pasta_videos} (vídeos normais)")
    print(f"  {PASTA_AMOSTRAS / 'video_exemplo_com_anomalia.mp4'} (vídeo com evento brusco)")
    print(f"  {pasta_audios} (áudios por classe)")


if __name__ == "__main__":
    gerar_todas_as_amostras()

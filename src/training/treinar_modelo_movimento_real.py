"""
Treino ALTERNATIVO do modelo de movimento, usando vídeos REAIS de postura
normal — complementar ao treino supervisionado com dados sintéticos de
`treinar_modelo_movimento.py` (que continua sendo o modelo usado pelo
sistema em produção, ver src/anomalies/movement_anomaly.py).

Mesma ideia do treino alternativo de sinais vitais com o PhysioNet
(treinar_modelo_vitais_physionet.py): dá pra treinar sem precisar rotular
anomalias manualmente, contanto que o modelo seja NÃO supervisionado
(IsolationForest, `contamination` baixo) e só veja exemplos "normais" — ele
aprende como é o padrão normal de movimento e sinaliza qualquer coisa fora
disso, sem nunca precisar de um exemplo real de queda/espasmo rotulado.

Rodar:  python -m src.training.treinar_modelo_movimento_real
        python -m src.training.treinar_modelo_movimento_real --pasta amostras/videos_normais

Os vídeos usados por padrão (fisioterapia simples, sintéticos mas com forma
humana reconhecível pelo MediaPipe) já vêm prontos em
`amostras/videos_normais/` — gerados por
`python -m src.dados.gerar_amostras_video_audio`.

Também é possível treinar com vídeos REAIS de fisioterapia, do dataset
Kaggle `toobasaeed11/physiotherapy` (ver
src/dados/baixar_kaggle_fisioterapia.py):

    python -m src.dados.baixar_kaggle_fisioterapia
    python -m src.training.treinar_modelo_movimento_real --pasta amostras/videos_kaggle_fisioterapia/raw

Ou apontar --pasta para qualquer outra pasta com vídeos reais próprios.
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import PASTA_AMOSTRAS
from src.training.conjunto_dados_movimento import TAMANHO_JANELA_FRAMES, extrair_atributos_da_janela
from src.training.utils import PASTA_MODELOS
from src.video.pose_analyzer import AnalisadorDePostura
from src.video.pose_features import NOMES_ATRIBUTOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("treino_movimento_real")

NOME_MODELO = "movimento_isolationforest_real"
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
COLUNAS_JANELA = [f"{nome}_media" for nome in NOMES_ATRIBUTOS] + [f"{nome}_desvio" for nome in NOMES_ATRIBUTOS]


def _listar_videos(pasta: Path) -> list[Path]:
    return sorted(caminho for caminho in pasta.rglob("*") if caminho.suffix.lower() in EXTENSOES_VIDEO)


def construir_conjunto_dados_real(pasta_videos: Path) -> tuple[list[dict], list[str]]:
    """
    Processa cada vídeo da pasta, desliza janelas de `TAMANHO_JANELA_FRAMES`
    frames e extrai os mesmos 16 atributos (média/desvio das 8 features de
    pose_features.py) usados no treino sintético — assim o modelo real fica
    compatível com o mesmo pipeline de inferência.
    """
    videos = _listar_videos(pasta_videos)
    if not videos:
        raise SystemExit(
            f"Nenhum vídeo encontrado em {pasta_videos}. Rode primeiro "
            "'python -m src.dados.gerar_amostras_video_audio' para gerar os vídeos de exemplo, "
            "ou aponte --pasta para uma pasta com vídeos reais próprios."
        )

    analisador = AnalisadorDePostura()
    janelas = []
    for indice, video in enumerate(videos, start=1):
        logger.info("[%d/%d] Extraindo postura de %s...", indice, len(videos), video.name)
        sequencia_pontos = analisador.extrair_pontos_completos(str(video))
        for fim in range(TAMANHO_JANELA_FRAMES, len(sequencia_pontos) + 1):
            janela = sequencia_pontos[fim - TAMANHO_JANELA_FRAMES:fim]
            janelas.append(extrair_atributos_da_janela(janela))

    return janelas, [str(video.relative_to(pasta_videos.parent.parent)) for video in videos]


def treinar(pasta_videos: Path | None = None, contaminacao: float = 0.05):
    pasta_videos = pasta_videos or (PASTA_AMOSTRAS / "videos_normais")

    logger.info("1/4 - Extraindo postura dos vídeos em '%s'...", pasta_videos)
    janelas, videos_usados = construir_conjunto_dados_real(pasta_videos)
    if len(janelas) < 20:
        raise SystemExit(f"Poucas janelas válidas ({len(janelas)}). Use vídeos mais longos ou em maior número.")

    import pandas as pd
    X = pd.DataFrame(janelas)[COLUNAS_JANELA]
    logger.info("Conjunto de dados real construído: %d janelas de %d vídeo(s).", len(X), len(videos_usados))

    logger.info("2/4 - Treinando IsolationForest (não supervisionado, só vê exemplos normais)...")
    modelo = IsolationForest(n_estimators=250, contamination=contaminacao, random_state=42)
    modelo.fit(X)

    logger.info("3/4 - Aplicando o modelo sobre a própria série para inspecionar o resultado...")
    previsao = modelo.predict(X)
    n_fora_do_padrao = int((previsao == -1).sum())
    logger.info("Janelas fora do padrão na própria série de treino: %d de %d (%.1f%%).",
                n_fora_do_padrao, len(X), 100 * n_fora_do_padrao / len(X))

    logger.info("4/4 - Salvando modelo...")
    caminho_modelo = PASTA_MODELOS / f"{NOME_MODELO}.joblib"
    joblib.dump({"modelo": modelo, "atributos": COLUNAS_JANELA, "videos_treino": videos_usados}, caminho_modelo)

    print(f"\nModelo (IsolationForest, vídeos reais) salvo em: {caminho_modelo}")
    print(f"Vídeos usados: {len(videos_usados)} | Janelas: {len(X)}")
    print(f"Janelas fora do padrão na própria série de treino: {n_fora_do_padrao}/{len(X)}")
    return modelo, {"n_videos": len(videos_usados), "n_janelas": len(X), "n_fora_do_padrao": n_fora_do_padrao}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasta", default=None, help="Pasta com vídeos normais (padrão: amostras/videos_normais)")
    parser.add_argument("--contaminacao", type=float, default=0.05)
    argumentos = parser.parse_args()

    caminho_pasta = Path(argumentos.pasta) if argumentos.pasta else None
    treinar(pasta_videos=caminho_pasta, contaminacao=argumentos.contaminacao)

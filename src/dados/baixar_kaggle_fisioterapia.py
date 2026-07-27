"""
Baixa o dataset REAL "Physiotherapy" do Kaggle (vídeos de sessões de
fisioterapia de pacientes de verdade) e prepara os arquivos para o treino
alternativo de movimento (src/training/treinar_modelo_movimento_real.py).

Dataset: `toobasaeed11/physiotherapy`
Página:  https://www.kaggle.com/datasets/toobasaeed11/physiotherapy

Complementa os vídeos sintéticos já incluídos em `amostras/videos_normais/`
(gerados por `src/dados/gerar_amostras_video_audio.py`) — aqueles garantem
que o pipeline funcione de ponta a ponta sem depender de internet; este
script traz vídeos REAIS de fisioterapia para quem quiser treinar/validar
com filmagens de verdade.

Rodar:  python -m src.dados.baixar_kaggle_fisioterapia

Requer uma conta no Kaggle com as condições do dataset aceitas, e
autenticação configurada (arquivo kaggle.json ou variáveis de ambiente
KAGGLE_USERNAME/KAGGLE_KEY) — ver instruções em
https://github.com/Kagglehub/kagglehub#authenticate. Sem isso, ou sem
internet, o script falha com uma mensagem clara em vez de travar.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from src.config import PASTA_AMOSTRAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("baixar_kaggle_fisioterapia")

IDENTIFICADOR_DATASET = "toobasaeed11/physiotherapy"
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
PASTA_PADRAO = "videos_kaggle_fisioterapia/raw"


def _copiar_arquivos(origem: Path, destino: Path) -> list[Path]:
    """Copia tudo que o kagglehub baixou (cache dele) para dentro do projeto, em amostras/."""
    destino.mkdir(parents=True, exist_ok=True)
    copiados = []
    for item in origem.rglob("*"):
        if not item.is_file():
            continue
        caminho_relativo = item.relative_to(origem)
        alvo = destino / caminho_relativo
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, alvo)
        copiados.append(alvo)
    return copiados


def _criar_manifesto(pasta_videos: Path, caminho_manifesto: Path) -> dict:
    """Registra quantos vídeos vieram, quais extensões, e a lista completa — útil para conferir o download."""
    videos = sorted(p for p in pasta_videos.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES_VIDEO)
    contagem_por_extensao: dict[str, int] = {}
    for video in videos:
        contagem_por_extensao[video.suffix.lower()] = contagem_por_extensao.get(video.suffix.lower(), 0) + 1

    manifesto = {
        "dataset": IDENTIFICADOR_DATASET,
        "fonte": "Kaggle",
        "pasta_videos": str(pasta_videos),
        "total_videos": len(videos),
        "videos_por_extensao": contagem_por_extensao,
        "videos": [str(v) for v in videos],
    }
    caminho_manifesto.parent.mkdir(parents=True, exist_ok=True)
    caminho_manifesto.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifesto


def baixar(pasta_destino: Path | None = None, forcar: bool = False) -> dict:
    """
    Baixa o dataset via `kagglehub`, copia os arquivos para dentro do
    projeto (`amostras/videos_kaggle_fisioterapia/raw/` por padrão) e gera
    um manifesto com o inventário de vídeos encontrados.
    """
    try:
        import kagglehub
    except ImportError as erro:
        raise SystemExit(
            "Pacote 'kagglehub' não instalado. Rode: pip install -r requirements.txt"
        ) from erro

    pasta_destino = pasta_destino or (PASTA_AMOSTRAS / PASTA_PADRAO)
    if forcar and pasta_destino.exists():
        shutil.rmtree(pasta_destino)

    logger.info("Baixando dataset Kaggle '%s'...", IDENTIFICADOR_DATASET)
    try:
        pasta_cache = Path(kagglehub.dataset_download(IDENTIFICADOR_DATASET))
    except Exception as erro:
        raise SystemExit(
            f"Não foi possível baixar o dataset '{IDENTIFICADOR_DATASET}' do Kaggle. "
            "Confirme que você aceitou as condições do dataset na página do Kaggle e configurou "
            "a autenticação (kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY). "
            f"Detalhe do erro: {erro}"
        ) from erro

    logger.info("Arquivos baixados em: %s", pasta_cache)
    copiados = _copiar_arquivos(pasta_cache, pasta_destino)
    manifesto = _criar_manifesto(pasta_destino, PASTA_AMOSTRAS / "videos_kaggle_fisioterapia" / "manifesto.json")

    print(f"\nArquivos copiados: {len(copiados)}")
    print(f"Vídeos encontrados: {manifesto['total_videos']}")
    print(f"Destino: {pasta_destino}")
    if not manifesto["total_videos"]:
        print("Aviso: nenhum vídeo com extensão reconhecida foi encontrado. Confira a estrutura baixada.")
    else:
        print(
            f"\nPara treinar o modelo de movimento com estes vídeos:\n"
            f"  python -m src.training.treinar_modelo_movimento_real --pasta {pasta_destino}"
        )

    return manifesto


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasta-destino", default=None, help=f"Destino dos vídeos (padrão: amostras/{PASTA_PADRAO})")
    parser.add_argument("--forcar", action="store_true", help="Apaga a pasta de destino antes de baixar de novo")
    argumentos = parser.parse_args()

    caminho_destino = Path(argumentos.pasta_destino) if argumentos.pasta_destino else None
    baixar(pasta_destino=caminho_destino, forcar=argumentos.forcar)

"""
Treino ALTERNATIVO do modelo de voz, usando áudios REAIS organizados em
pastas por classe — complementar ao treino supervisionado com atributos
sintéticos de `treinar_modelo_audio.py` (que continua sendo o modelo usado
pelo sistema em produção, ver src/audio/risco_fala.py).

Em vez de simular pitch/silêncio/energia como o treino sintético faz, este
script usa TRANSFER LEARNING de verdade: extrai o embedding do YAMNet
(rede já treinada no AudioSet completo, 2 milhões de clipes — ver
src/audio/audioset_eventos.py) de cada áudio real e treina um
RandomForestClassifier leve por cima desses embeddings. Isso aproveita tudo
que o YAMNet já aprendeu sobre som em geral, sem precisar de milhares de
áudios próprios para treinar do zero.

Rodar:  python -m src.training.treinar_modelo_audio_real
        python -m src.training.treinar_modelo_audio_real --pasta amostras/audio_por_classe

Os áudios usados por padrão (tons sintéticos simples, um jeito de exercitar
o pipeline de ponta a ponta) já vêm prontos em `amostras/audio_por_classe/`,
organizados em subpastas por classe:
    amostras/audio_por_classe/normal/*.wav
    amostras/audio_por_classe/alterado/*.wav

Gerados por `python -m src.dados.gerar_amostras_video_audio`. Também é
possível apontar --pasta para áudios reais próprios organizados da mesma
forma (uma subpasta por classe, pelo menos 2 classes).
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.audio.audioset_eventos import extrair_embedding
from src.config import PASTA_AMOSTRAS
from src.training.utils import PASTA_MODELOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("treino_audio_real")

NOME_MODELO = "audio_rf_embeddings_real"
EXTENSOES_AUDIO = {".wav", ".flac", ".ogg", ".mp3"}


def _coletar_arquivos_por_classe(pasta_base: Path) -> list[tuple[Path, str]]:
    if not pasta_base.exists():
        return []
    arquivos = []
    for pasta_classe in sorted(p for p in pasta_base.iterdir() if p.is_dir()):
        for audio in sorted(pasta_classe.rglob("*")):
            if audio.suffix.lower() in EXTENSOES_AUDIO:
                arquivos.append((audio, pasta_classe.name))
    return arquivos


def treinar(pasta_base: Path | None = None):
    pasta_base = pasta_base or (PASTA_AMOSTRAS / "audio_por_classe")

    logger.info("1/4 - Procurando áudios reais em '%s' (uma subpasta por classe)...", pasta_base)
    arquivos = _coletar_arquivos_por_classe(pasta_base)
    if len(arquivos) < 10:
        raise SystemExit(
            f"Encontrados {len(arquivos)} áudio(s) em {pasta_base}. Rode primeiro "
            "'python -m src.dados.gerar_amostras_video_audio' para gerar os áudios de exemplo, "
            "ou inclua pelo menos 10 áudios reais distribuídos em 2+ subpastas (uma por classe)."
        )

    logger.info("2/4 - Extraindo embeddings do YAMNet (1024 dimensões por áudio)...")
    X, y = [], []
    for indice, (audio, classe) in enumerate(arquivos, start=1):
        logger.info("[%d/%d] %s: %s", indice, len(arquivos), classe, audio.name)
        try:
            X.append(extrair_embedding(str(audio)))
            y.append(classe)
        except Exception as erro:
            logger.warning("Ignorado (%s): %s", audio.name, erro)

    classes_encontradas = sorted(set(y))
    if len(classes_encontradas) < 2:
        raise SystemExit(f"São necessárias pelo menos 2 classes com áudio válido. Encontrada(s): {classes_encontradas}")

    X, y = np.asarray(X), np.asarray(y)
    logger.info("3/4 - Separando treino/teste e treinando RandomForestClassifier...")
    contagens = np.unique(y, return_counts=True)[1]
    estratificar = y if contagens.min() >= 2 else None
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=estratificar)

    modelo = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    modelo.fit(X_treino, y_treino)

    from sklearn.metrics import classification_report
    y_previsto = modelo.predict(X_teste)
    relatorio = classification_report(y_teste, y_previsto, zero_division=0)
    logger.info("Relatório de classificação (conjunto de teste):\n%s", relatorio)

    logger.info("4/4 - Salvando modelo...")
    caminho_modelo = PASTA_MODELOS / f"{NOME_MODELO}.joblib"
    joblib.dump({
        "modelo": modelo,
        "classes": classes_encontradas,
        "n_audios_treino": len(X),
        "origem": "embeddings YAMNet (transfer learning sobre o AudioSet)",
    }, caminho_modelo)

    print(f"\nModelo (RandomForest sobre embeddings YAMNet) salvo em: {caminho_modelo}")
    print(f"Áudios usados: {len(X)} | Classes: {classes_encontradas}")
    return modelo, {"n_audios": len(X), "classes": classes_encontradas}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasta", default=None, help="Pasta com subpastas por classe (padrão: amostras/audio_por_classe)")
    argumentos = parser.parse_args()

    caminho_pasta = Path(argumentos.pasta) if argumentos.pasta else None
    treinar(pasta_base=caminho_pasta)

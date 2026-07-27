"""
Classificação de eventos sonoros com YAMNet — rede treinada pelo Google
DIRETAMENTE no AudioSet (o dataset sugerido no enunciado do desafio).

Isto resolve uma limitação que deixamos documentada nas primeiras versões
deste projeto: o AudioSet não distribui os arquivos de áudio para download
(só metadados de vídeos do YouTube), então não dava para treinar do zero
com ele. O YAMNet contorna esse problema: é um modelo já treinado no
AudioSet completo (2 milhões de clipes, 521 classes de som) e distribuído
publicamente via TensorFlow Hub — ou seja, usamos o AudioSet de verdade,
só que através do modelo pronto, em vez de baixar o dataset bruto.

Uso: identificar, no áudio de uma consulta, a presença de sons clinicamente
relevantes (tosse, respiração ofegante, engasgo, choro, gemido etc.) que
complementam a transcrição de fala. Além da classificação direta (521
classes do AudioSet), também expomos o EMBEDDING interno do YAMNet
(`extrair_embedding`) — um vetor de 1024 números que resume o áudio — para
servir de feature de entrada a um classificador leve treinado com áudios
reais próprios (ver src/training/treinar_modelo_audio_real.py). É a técnica
de "transfer learning": aproveitamos o que o YAMNet já aprendeu em 2 milhões
de clipes do AudioSet, sem precisar treinar uma rede de áudio do zero.
"""

import csv
import logging

import numpy as np

logger = logging.getLogger(__name__)

URL_MODELO_YAMNET = "https://tfhub.dev/google/yamnet/1"

# Subconjunto das 521 classes do AudioSet com relevância clínica direta.
CLASSES_RELEVANTES = {
    "Speech", "Cough", "Sneeze", "Breathing", "Wheeze", "Snoring", "Gasp",
    "Throat clearing", "Crying, sobbing", "Screaming", "Groan", "Vomiting",
    "Hiccup", "Sniff", "Pant", "Baby cry, infant cry", "Choking",
}

_modelo_yamnet = None  # cache em memória: carregar o YAMNet é caro, então fazemos isso 1x por processo


def _carregar_yamnet():
    """Carrega o YAMNet do TensorFlow Hub uma única vez por processo (fica em cache depois)."""
    global _modelo_yamnet
    if _modelo_yamnet is None:
        import tensorflow_hub as hub

        logger.info("Carregando YAMNet (modelo treinado no AudioSet)...")
        _modelo_yamnet = hub.load(URL_MODELO_YAMNET)
    return _modelo_yamnet


def classificar_eventos_sonoros(caminho_audio: str, top_k: int = 8) -> list[dict]:
    """
    Roda o YAMNet sobre um áudio e devolve as `top_k` classes de som mais
    prováveis, marcando quais têm relevância clínica.

    Requer `tensorflow` e `tensorflow_hub` instalados (ver requirements.txt)
    e acesso à internet na primeira execução (o modelo é baixado e fica em
    cache local do TensorFlow Hub nas execuções seguintes).
    """
    import librosa
    import tensorflow as tf

    modelo = _carregar_yamnet()

    onda, _ = librosa.load(caminho_audio, sr=16000, mono=True)
    pontuacoes, _, _ = modelo(tf.convert_to_tensor(onda, dtype=tf.float32))
    pontuacoes_medias = pontuacoes.numpy().mean(axis=0)

    caminho_classes = modelo.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(caminho_classes) as arquivo:
        nomes_classes = [linha["display_name"] for linha in csv.DictReader(arquivo)]

    indices_top = np.argsort(pontuacoes_medias)[::-1][:top_k]
    resultado = [
        {
            "classe": nomes_classes[indice],
            "pontuacao": round(float(pontuacoes_medias[indice]), 4),
            "relevante_clinicamente": nomes_classes[indice] in CLASSES_RELEVANTES,
        }
        for indice in indices_top
    ]

    logger.info("YAMNet: %d eventos sonoros identificados, %d com relevância clínica.",
                len(resultado), sum(1 for e in resultado if e["relevante_clinicamente"]))
    return resultado


def extrair_embedding(caminho_audio: str) -> np.ndarray:
    """
    Devolve o embedding médio do YAMNet (1024 dimensões) para um áudio —
    uma "impressão digital" numérica do som, aprendida sobre o AudioSet
    completo. Usado como feature de entrada para treinar um classificador
    leve (RandomForest) com áudios reais próprios, em vez das 521 classes
    fixas do AudioSet — ver src/training/treinar_modelo_audio_real.py.
    """
    import librosa
    import tensorflow as tf

    modelo = _carregar_yamnet()
    onda, _ = librosa.load(caminho_audio, sr=16000, mono=True)
    _, embeddings, _ = modelo(tf.convert_to_tensor(onda, dtype=tf.float32))
    return embeddings.numpy().mean(axis=0).astype(np.float32)

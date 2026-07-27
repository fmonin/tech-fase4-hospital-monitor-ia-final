"""
Conjunto de dados rotulado para treinar o classificador de anomalias de
MOVIMENTO, usado sobre a sequência de pontos-chave (33 pontos do corpo,
MediaPipe) extraída por src/video/pose_analyzer.py a partir de um vídeo
real.

Como executar:
1) Treino completo do modelo que usa este módulo:
    python -m src.training.treinar_modelo_movimento

2) Rodar este módulo diretamente para gerar o conjunto sintético:
    python -m src.training.conjunto_dados_movimento

3) Gerar com parâmetros customizados:
    python -m src.training.conjunto_dados_movimento --n-por-classe 800 --semente 123

4) Gerar e salvar CSV com atributos + rótulo `anomalo`:
    python -m src.training.conjunto_dados_movimento --saida reports/conjunto_movimento.csv

5) Teste rápido via linha única (sem CLI):
    python -c "from src.training.conjunto_dados_movimento import construir_conjunto_dados; X,y=construir_conjunto_dados(); print(X.shape, y.shape)"

Mesma lógica do conjunto de dados de vitais: geramos janelas sintéticas de
postura normal (parada, com pequeno balanço) e janelas com uma anomalia
postural injetada (assimetria de ombro/quadril, tronco inclinado, joelho
travado, movimento brusco) — o rótulo nasce do próprio processo de geração,
então é 100% confiável para treino supervisionado.

Cada exemplo é uma JANELA de frames, resumida em média e desvio-padrão dos 8
atributos clínicos de src/video/pose_features.py (16 colunas ao todo) — o
mesmo resumo (`pose_features.agregar_janela`) usado em produção, para que
treino e inferência nunca fiquem fora de sincronia.
"""

import argparse

import numpy as np
import pandas as pd

from src.video.pose_features import agregar_janela, extrair_atributos_pose, pontos_base_neutros

TAMANHO_JANELA_FRAMES = 15

# Cada tipo de anomalia mexe num "canal" diferente do esqueleto, para que o
# modelo aprenda a reconhecer os vários padrões posturais citados no PDF do
# desafio (queda, espasmo, imobilidade fora do esperado etc.), não só um.
_TIPOS_DE_ANOMALIA = [
    "assimetria_ombro",
    "assimetria_quadril",
    "tronco_inclinado",
    "joelho_travado",
    "movimento_brusco",
]


def _gerar_janela_pontos(rng: np.random.Generator, n_frames: int, anomala: bool) -> np.ndarray:
    """Gera uma janela (n_frames, 33, 3) de postura normal ou com anomalia injetada."""
    base = pontos_base_neutros()
    janela = np.repeat(base[np.newaxis, :, :], n_frames, axis=0)
    janela[:, :, :2] += rng.normal(0, 0.005, (n_frames, 33, 2))  # jitter natural do tracking

    if not anomala:
        return janela

    tipo = rng.choice(_TIPOS_DE_ANOMALIA)
    inicio = rng.integers(0, max(1, n_frames - 5))
    duracao = int(rng.integers(3, min(6, n_frames - inicio) + 1))
    fim = inicio + duracao
    rampa = np.linspace(0, 1, duracao)

    if tipo == "assimetria_ombro":
        janela[inicio:fim, 12, 1] += rampa * rng.uniform(0.15, 0.25)  # ombro direito cai
    elif tipo == "assimetria_quadril":
        janela[inicio:fim, 24, 1] += rampa * rng.uniform(0.12, 0.20)  # quadril direito cai
    elif tipo == "tronco_inclinado":
        deslocamento = rampa * rng.uniform(0.15, 0.25)
        janela[inicio:fim, [11, 12], 0] += deslocamento[:, np.newaxis]  # ombros deslocam lateralmente
    elif tipo == "joelho_travado":
        janela[inicio:fim, 26, 0] += rampa * rng.uniform(0.15, 0.22)  # tornozelo/joelho direito rígido
        janela[inicio:fim, 28, 0] += rampa * rng.uniform(0.15, 0.22)
    elif tipo == "movimento_brusco":
        janela[inicio:fim, :, 0] += rampa[:, np.newaxis] * rng.uniform(0.20, 0.35)  # corpo inteiro se desloca rápido

    return janela


def extrair_atributos_da_janela(janela_pontos: np.ndarray) -> dict:
    """
    Roda `extrair_atributos_pose` frame a frame e resume a janela em
    média/desvio (usada tanto para gerar o conjunto sintético abaixo quanto
    pelo treino alternativo com vídeos reais,
    src/training/treinar_modelo_movimento_real.py).
    """
    atributos_por_frame = []
    anterior = None
    for frame in janela_pontos:
        atributos_por_frame.append(extrair_atributos_pose(frame, pontos_anteriores=anterior))
        anterior = frame
    return agregar_janela(atributos_por_frame)


def construir_conjunto_dados(n_por_classe: int = 500, semente: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Gera um conjunto balanceado de janelas de postura normais x anômalas."""
    rng = np.random.default_rng(semente)
    linhas, rotulos = [], []

    for _ in range(n_por_classe):
        janela = _gerar_janela_pontos(rng, TAMANHO_JANELA_FRAMES, anomala=False)
        linhas.append(extrair_atributos_da_janela(janela))
        rotulos.append(0)

    for _ in range(n_por_classe):
        janela = _gerar_janela_pontos(rng, TAMANHO_JANELA_FRAMES, anomala=True)
        linhas.append(extrair_atributos_da_janela(janela))
        rotulos.append(1)

    X = pd.DataFrame(linhas)
    y = pd.Series(rotulos, name="anomalo")

    # Introduz ambiguidade controlada para evitar métricas artificiais de 100%.
    proporcao_ruido_rotulo = 0.03
    n_ruido = max(1, int(len(y) * proporcao_ruido_rotulo))
    indices_ruido = rng.choice(len(y), size=n_ruido, replace=False)
    y.iloc[indices_ruido] = 1 - y.iloc[indices_ruido]

    return X, y


def _main():
    parser = argparse.ArgumentParser(description="Gera conjunto de dados sintético de movimento (postura normal x anômala).")
    parser.add_argument("--n-por-classe", type=int, default=500, help="Quantidade de exemplos por classe (normal e anômalo).")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória para reprodução.")
    parser.add_argument(
        "--saida",
        default=None,
        help="Caminho opcional para salvar CSV com atributos + coluna 'anomalo'. Ex.: reports/conjunto_movimento.csv",
    )
    args = parser.parse_args()

    X, y = construir_conjunto_dados(n_por_classe=args.n_por_classe, semente=args.semente)
    print(f"Conjunto gerado: X={X.shape}, y={y.shape}")
    print(f"Distribuição de classes: {y.value_counts().to_dict()}")

    if args.saida:
        df = X.copy()
        df["anomalo"] = y.values
        df.to_csv(args.saida, index=False)
        print(f"Arquivo salvo em: {args.saida}")


if __name__ == "__main__":
    _main()

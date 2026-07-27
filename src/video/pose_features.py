"""
Extração de atributos clínicos explicáveis a partir dos 33 pontos-chave do
corpo (MediaPipe Pose): assimetria de ombros/quadril, inclinação do tronco,
ângulos de joelhos e cotovelos, e velocidade de movimento.

Isto substitui a ideia mais simples de "acompanhar só um ponto-chave" por
algo mais próximo do que um fisioterapeuta de verdade observaria: não é só
"o pulso se moveu rápido", é "o ombro está mais alto que o outro" ou "o
joelho direito dobra muito menos que o esquerdo" — atributos que fazem
sentido clínico por si só, mesmo antes de qualquer modelo de anomalia
entrar em cena.
"""

import numpy as np

NOMES_ATRIBUTOS = [
    "assimetria_ombros",
    "assimetria_quadril",
    "inclinacao_tronco_graus",
    "angulo_joelho_esquerdo_graus",
    "angulo_joelho_direito_graus",
    "angulo_cotovelo_esquerdo_graus",
    "angulo_cotovelo_direito_graus",
    "velocidade_movimento",
]

# Índices dos pontos-chave do MediaPipe Pose (mp.solutions.pose.PoseLandmark).
_OMBRO_ESQUERDO, _OMBRO_DIREITO = 11, 12
_COTOVELO_ESQUERDO, _COTOVELO_DIREITO = 13, 14
_PUNHO_ESQUERDO, _PUNHO_DIREITO = 15, 16
_QUADRIL_ESQUERDO, _QUADRIL_DIREITO = 23, 24
_JOELHO_ESQUERDO, _JOELHO_DIREITO = 25, 26
_TORNOZELO_ESQUERDO, _TORNOZELO_DIREITO = 27, 28


def _valido(ponto: np.ndarray, confianca_minima: float = 0.5) -> bool:
    """Um ponto é válido se o MediaPipe está razoavelmente confiante nele."""
    return len(ponto) >= 3 and np.isfinite(ponto[:2]).all() and ponto[2] >= confianca_minima


def _angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo em graus no vértice B, formado pelos segmentos B->A e B->C."""
    if not (_valido(a) and _valido(b) and _valido(c)):
        return float("nan")
    ba, bc = a[:2] - b[:2], c[:2] - b[:2]
    denominador = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominador == 0:
        return float("nan")
    cosseno = np.clip(np.dot(ba, bc) / denominador, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosseno)))


def extrair_atributos_pose(pontos: np.ndarray, pontos_anteriores: np.ndarray | None = None) -> dict:
    """
    Recebe os pontos-chave de um frame — array (33, 3): x, y normalizados
    (0-1) + confiança (visibility) do MediaPipe — e devolve os 8 atributos
    explicáveis de `NOMES_ATRIBUTOS`. Pontos ausentes/de baixa confiança
    geram `nan` nos atributos que dependem deles (tratado por quem consome).
    """
    valores = {nome: float("nan") for nome in NOMES_ATRIBUTOS}
    if pontos is None or len(pontos) < 29:
        return valores

    ombro_e, ombro_d = pontos[_OMBRO_ESQUERDO], pontos[_OMBRO_DIREITO]
    cotovelo_e, cotovelo_d = pontos[_COTOVELO_ESQUERDO], pontos[_COTOVELO_DIREITO]
    punho_e, punho_d = pontos[_PUNHO_ESQUERDO], pontos[_PUNHO_DIREITO]
    quadril_e, quadril_d = pontos[_QUADRIL_ESQUERDO], pontos[_QUADRIL_DIREITO]
    joelho_e, joelho_d = pontos[_JOELHO_ESQUERDO], pontos[_JOELHO_DIREITO]
    tornozelo_e, tornozelo_d = pontos[_TORNOZELO_ESQUERDO], pontos[_TORNOZELO_DIREITO]

    if _valido(ombro_e) and _valido(ombro_d):
        valores["assimetria_ombros"] = float(abs(ombro_e[1] - ombro_d[1]))
    if _valido(quadril_e) and _valido(quadril_d):
        valores["assimetria_quadril"] = float(abs(quadril_e[1] - quadril_d[1]))

    if all(_valido(p) for p in (ombro_e, ombro_d, quadril_e, quadril_d)):
        centro_ombros = (ombro_e[:2] + ombro_d[:2]) / 2
        centro_quadril = (quadril_e[:2] + quadril_d[:2]) / 2
        vetor_tronco = centro_ombros - centro_quadril
        # 0° = tronco perfeitamente vertical; valores altos = inclinação lateral.
        valores["inclinacao_tronco_graus"] = float(abs(np.degrees(np.arctan2(vetor_tronco[0], -vetor_tronco[1]))))

    valores["angulo_joelho_esquerdo_graus"] = _angulo(quadril_e, joelho_e, tornozelo_e)
    valores["angulo_joelho_direito_graus"] = _angulo(quadril_d, joelho_d, tornozelo_d)
    valores["angulo_cotovelo_esquerdo_graus"] = _angulo(ombro_e, cotovelo_e, punho_e)
    valores["angulo_cotovelo_direito_graus"] = _angulo(ombro_d, cotovelo_d, punho_d)

    if pontos_anteriores is not None and pontos_anteriores.shape == pontos.shape:
        validos = np.array([_valido(p) and _valido(q) for p, q in zip(pontos, pontos_anteriores)])
        if validos.any():
            deslocamentos = np.linalg.norm(pontos[validos, :2] - pontos_anteriores[validos, :2], axis=1)
            valores["velocidade_movimento"] = float(np.mean(deslocamentos))

    return valores


LIMIAR_ASSIMETRIA_OMBROS = 0.10
LIMIAR_ASSIMETRIA_QUADRIL = 0.08
LIMIAR_INCLINACAO_TRONCO = 20.0
LIMIAR_DIFERENCA_JOELHOS = 25.0


def motivos_da_regra(atributos: dict) -> list[str]:
    """
    Além do modelo treinado, aplicamos regras simples e explicáveis sobre
    os mesmos atributos — úteis tanto como camada de apoio (concordância
    entre regra e modelo aumenta a confiança do alerta) quanto como
    fallback caso o modelo treinado ainda não exista.
    """
    motivos = []
    assimetria_ombros = atributos.get("assimetria_ombros", float("nan"))
    assimetria_quadril = atributos.get("assimetria_quadril", float("nan"))
    inclinacao = atributos.get("inclinacao_tronco_graus", float("nan"))
    joelho_e = atributos.get("angulo_joelho_esquerdo_graus", float("nan"))
    joelho_d = atributos.get("angulo_joelho_direito_graus", float("nan"))

    if np.isfinite(assimetria_ombros) and assimetria_ombros > LIMIAR_ASSIMETRIA_OMBROS:
        motivos.append("assimetria de ombros")
    if np.isfinite(assimetria_quadril) and assimetria_quadril > LIMIAR_ASSIMETRIA_QUADRIL:
        motivos.append("assimetria de quadril")
    if np.isfinite(inclinacao) and inclinacao > LIMIAR_INCLINACAO_TRONCO:
        motivos.append("inclinação excessiva do tronco")
    if np.isfinite(joelho_e) and np.isfinite(joelho_d) and abs(joelho_e - joelho_d) > LIMIAR_DIFERENCA_JOELHOS:
        motivos.append("diferença acentuada entre os joelhos")

    return motivos


def agregar_janela(atributos_por_frame: list[dict]) -> dict:
    """
    Resume uma janela de frames (cada um já passado por `extrair_atributos_pose`)
    em média e desvio-padrão de cada um dos 8 atributos — o mesmo formato é
    usado tanto para gerar o conjunto de dados de treino
    (src/training/conjunto_dados_movimento.py) quanto para classificar
    janelas reais em produção (src/anomalies/movement_anomaly.py), então os
    dois lados nunca ficam "fora de sincronia" sobre quais colunas o modelo
    espera.

    `nan` (ponto ausente/baixa confiança no frame) é ignorado no cálculo;
    se um atributo não tiver nenhum valor válido na janela inteira, o
    resultado fica em 0.0 (janela sem informação, mas sem quebrar o modelo).
    """
    resumo = {}
    for nome in NOMES_ATRIBUTOS:
        valores = np.array([frame[nome] for frame in atributos_por_frame], dtype=float)
        validos = valores[np.isfinite(valores)]
        resumo[f"{nome}_media"] = float(np.mean(validos)) if len(validos) else 0.0
        resumo[f"{nome}_desvio"] = float(np.std(validos)) if len(validos) else 0.0
    return resumo


def pontos_base_neutros() -> np.ndarray:
    """
    Um esqueleto (33, 3) parado, simétrico e de pé, usado como ponto de
    partida para gerar sequências sintéticas (dados de treino e dados de
    demonstração) — assim as duas partes do projeto concordam sobre a
    aparência de uma postura "normal", antes de qualquer ruído ou anomalia
    ser adicionado por cima.
    """
    pontos = np.zeros((33, 3))
    pontos[:, 2] = 0.9  # confiança alta em todos os pontos, por padrão
    pontos[_OMBRO_ESQUERDO] = [0.4, 0.30, 0.9]
    pontos[_OMBRO_DIREITO] = [0.6, 0.30, 0.9]
    pontos[_COTOVELO_ESQUERDO] = [0.35, 0.45, 0.9]
    pontos[_COTOVELO_DIREITO] = [0.65, 0.45, 0.9]
    pontos[_PUNHO_ESQUERDO] = [0.33, 0.60, 0.9]
    pontos[_PUNHO_DIREITO] = [0.67, 0.60, 0.9]
    pontos[_QUADRIL_ESQUERDO] = [0.42, 0.55, 0.9]
    pontos[_QUADRIL_DIREITO] = [0.58, 0.55, 0.9]
    pontos[_JOELHO_ESQUERDO] = [0.42, 0.75, 0.9]
    pontos[_JOELHO_DIREITO] = [0.58, 0.75, 0.9]
    pontos[_TORNOZELO_ESQUERDO] = [0.42, 0.95, 0.9]
    pontos[_TORNOZELO_DIREITO] = [0.58, 0.95, 0.9]
    return pontos

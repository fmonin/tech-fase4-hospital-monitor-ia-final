"""
Detecção de anomalias de movimento a partir da sequência de pontos-chave
(33 pontos do corpo, extraídos pelo MediaPipe em
src/video/pose_analyzer.py).

Duas camadas, na mesma linha do detector de sinais vitais:

1. Modelo treinado (RandomForest, ver src/training/treinar_modelo_movimento.py):
   olha para janelas de `TAMANHO_JANELA_FRAMES` frames e classifica cada
   janela como normal ou anômala, usando a média/desvio dos 8 atributos
   clínicos explicáveis de src/video/pose_features.py (assimetria de
   ombros/quadril, ângulos de joelho/cotovelo, inclinação de tronco,
   velocidade). É o caminho usado quando o vídeo enviado é real.
2. Regra por limiar (sem treino): aplica `pose_features.motivos_da_regra`
   frame a frame e sinaliza janelas onde a maioria dos frames tem algum
   motivo — serve de fallback caso o modelo treinado ainda não exista.

As duas camadas convergem para o mesmo formato de evento, incluindo os
`motivos` (mesmo quando é o modelo treinado que decide), para que o alerta
final seja sempre explicável.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.training.utils import carregar_modelo_treinado
from src.video.pose_features import agregar_janela, extrair_atributos_pose, motivos_da_regra

logger = logging.getLogger(__name__)

TAMANHO_JANELA_FRAMES = 15  # precisa bater com src/training/conjunto_dados_movimento.py
NOME_MODELO = "movimento_rf"
PROPORCAO_MINIMA_FRAMES_COM_MOTIVO = 0.5  # >= metade dos frames da janela com algum motivo, no fallback


class DetectorAnomaliaMovimento:
    def __init__(self):
        self._modelo_treinado = carregar_modelo_treinado(NOME_MODELO)
        if self._modelo_treinado is None:
            logger.warning(
                "Modelo treinado '%s' não encontrado em models/. Usando regra por limiar (mais simples). "
                "Rode: python -m src.training.treinar_modelo_movimento",
                NOME_MODELO,
            )

    def detectar(self, sequencia_pontos: np.ndarray) -> list[dict]:
        """
        Recebe um array (n_frames, 33, 3) de pontos-chave (ver
        `AnalisadorDePostura.extrair_pontos_completos`) e devolve os
        eventos anômalos, deslizando uma janela de `TAMANHO_JANELA_FRAMES`
        frames por vez.
        """
        if len(sequencia_pontos) < TAMANHO_JANELA_FRAMES:
            logger.info("Vídeo curto demais para análise de movimento confiável.")
            return []

        atributos_por_frame = self._extrair_atributos_por_frame(sequencia_pontos)

        if self._modelo_treinado is not None:
            eventos = self._detectar_com_modelo_treinado(atributos_por_frame)
        else:
            eventos = self._detectar_com_regra_por_limiar(atributos_por_frame)

        logger.info("Detecção de anomalias de movimento: %d evento(s) encontrado(s).", len(eventos))
        return eventos

    @staticmethod
    def _extrair_atributos_por_frame(sequencia_pontos: np.ndarray) -> list[dict]:
        """Aplica `extrair_atributos_pose` a cada frame, passando o anterior para calcular velocidade."""
        atributos_por_frame = []
        anterior = None
        for frame in sequencia_pontos:
            atributos_por_frame.append(extrair_atributos_pose(frame, pontos_anteriores=anterior))
            anterior = frame
        return atributos_por_frame

    def _detectar_com_modelo_treinado(self, atributos_por_frame: list[dict]) -> list[dict]:
        eventos = []
        for fim_janela in range(TAMANHO_JANELA_FRAMES, len(atributos_por_frame) + 1):
            inicio_janela = fim_janela - TAMANHO_JANELA_FRAMES
            janela = atributos_por_frame[inicio_janela:fim_janela]

            atributos_agregados = pd.DataFrame([agregar_janela(janela)])
            previsto = self._modelo_treinado.predict(atributos_agregados)[0]
            if previsto != 1:
                continue
            probabilidade = self._modelo_treinado.predict_proba(atributos_agregados)[0][1]
            severidade = "alta" if probabilidade >= 0.8 else "media"
            motivos = self._motivos_da_janela(janela)
            if not motivos:
                motivos = self._motivos_modelo_por_atributos(atributos_agregados)

            eventos.append({
                "frame": fim_janela - 1,
                "janela": [inicio_janela, fim_janela - 1],
                "motivos": motivos,
                "confianca_modelo": round(float(probabilidade), 3),
                "severidade": severidade,
            })
        return eventos

    def _detectar_com_regra_por_limiar(self, atributos_por_frame: list[dict]) -> list[dict]:
        eventos = []
        for fim_janela in range(TAMANHO_JANELA_FRAMES, len(atributos_por_frame) + 1):
            inicio_janela = fim_janela - TAMANHO_JANELA_FRAMES
            janela = atributos_por_frame[inicio_janela:fim_janela]
            motivos = self._motivos_da_janela(janela)

            frames_com_motivo = sum(1 for frame in janela if motivos_da_regra(frame))
            if frames_com_motivo / len(janela) < PROPORCAO_MINIMA_FRAMES_COM_MOTIVO:
                continue

            if not motivos:
                motivos = self._motivos_modelo_por_atributos(pd.DataFrame([agregar_janela(janela)]))

            eventos.append({
                "frame": fim_janela - 1,
                "janela": [inicio_janela, fim_janela - 1],
                "motivos": motivos,
                "proporcao_frames_com_motivo": round(frames_com_motivo / len(janela), 2),
                "severidade": "alta" if frames_com_motivo / len(janela) >= 0.8 else "media",
            })
        return eventos

    @staticmethod
    def _motivos_da_janela(janela_atributos: list[dict]) -> list[str]:
        """Une (sem repetir) os motivos de regra encontrados em qualquer frame da janela."""
        motivos = []
        for frame in janela_atributos:
            for motivo in motivos_da_regra(frame):
                if motivo not in motivos:
                    motivos.append(motivo)
        return motivos

    def _motivos_modelo_por_atributos(self, atributos_agregados: pd.DataFrame) -> list[str]:
        """
        Gera explicações a partir dos próprios atributos usados pelo classificador.
        Se houver importâncias no modelo, prioriza os atributos com maior peso.
        """
        if atributos_agregados.empty:
            return ["janela anômala detectada pelo classificador de movimento"]

        linha = atributos_agregados.iloc[0]
        importancias = {}
        if hasattr(self._modelo_treinado, "feature_importances_"):
            nomes = list(atributos_agregados.columns)
            pesos = list(getattr(self._modelo_treinado, "feature_importances_"))
            importancias = {nome: peso for nome, peso in zip(nomes, pesos)}

        # Pontua cada feature por |valor| ponderado pela importância do modelo.
        pontuacoes = []
        for nome, valor in linha.items():
            try:
                valor_num = float(valor)
            except (TypeError, ValueError):
                continue
            peso = float(importancias.get(nome, 1.0))
            pontuacoes.append((nome, abs(valor_num) * max(peso, 1e-6), valor_num, peso))

        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        top = pontuacoes[:4]

        explicacoes = []
        for nome, _score, valor, _peso in top:
            base, _, tipo = nome.rpartition("_")
            tipo_legivel = "média" if tipo == "media" else "variabilidade" if tipo == "desvio" else tipo

            if base == "assimetria_ombros":
                explicacoes.append(f"assimetria de ombros elevada ({tipo_legivel}={valor:.3f})")
            elif base == "assimetria_quadril":
                explicacoes.append(f"assimetria de quadril elevada ({tipo_legivel}={valor:.3f})")
            elif base == "inclinacao_tronco_graus":
                explicacoes.append(f"inclinação do tronco alterada ({tipo_legivel}={valor:.2f} graus)")
            elif base == "angulo_joelho_esquerdo_graus":
                explicacoes.append(f"ângulo do joelho esquerdo fora do padrão ({tipo_legivel}={valor:.2f} graus)")
            elif base == "angulo_joelho_direito_graus":
                explicacoes.append(f"ângulo do joelho direito fora do padrão ({tipo_legivel}={valor:.2f} graus)")
            elif base == "angulo_cotovelo_esquerdo_graus":
                explicacoes.append(f"ângulo do cotovelo esquerdo fora do padrão ({tipo_legivel}={valor:.2f} graus)")
            elif base == "angulo_cotovelo_direito_graus":
                explicacoes.append(f"ângulo do cotovelo direito fora do padrão ({tipo_legivel}={valor:.2f} graus)")
            elif base == "velocidade_movimento":
                explicacoes.append(f"velocidade de movimento alterada ({tipo_legivel}={valor:.4f})")

        # Remove duplicatas preservando ordem.
        dedup = list(dict.fromkeys(explicacoes))
        if dedup:
            return dedup[:3]

        # Fallback técnico, ainda baseado em feature real.
        if top:
            nome, _score, valor, _peso = top[0]
            return [f"atributo dominante do modelo: {nome}={valor:.4f}"]
        return ["janela anômala detectada pelo classificador de movimento"]

"""
Detecção de anomalias na evolução das prescrições médicas de um paciente.

Aqui optamos deliberadamente por regras simples e explicáveis em vez de um
modelo estatístico: mudanças de prescrição são eventos raros e discretos
(não uma série contínua), e a equipe médica precisa entender EXATAMENTE por
que algo foi sinalizado — "a dose subiu 300% de um dia para o outro" é uma
explicação útil; "o modelo achou estranho" não é.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LIMITE_VARIACAO_DOSE = 0.5   # sinaliza se a dose mudar mais que 50%
LIMITE_VARIACAO_FREQUENCIA = 0.5


class DetectorAnomaliaPrescricao:
    def detectar(self, historico_prescricoes: pd.DataFrame) -> list[dict]:
        """
        Espera um DataFrame ordenado por data, com colunas:
        data | medicamento | dose_mg | frequencia_horas

        Compara cada prescrição de um medicamento com a prescrição anterior
        do MESMO medicamento e sinaliza variações abruptas de dose ou
        frequência de administração.
        """
        eventos = []
        df = historico_prescricoes.sort_values("data")

        for medicamento, grupo in df.groupby("medicamento"):
            grupo = grupo.sort_values("data").reset_index(drop=True)
            for i in range(1, len(grupo)):
                anterior, atual = grupo.iloc[i - 1], grupo.iloc[i]
                eventos += self._comparar_prescricoes(medicamento, anterior, atual)

        eventos.sort(key=lambda e: e["data"])
        logger.info("Detecção de anomalias em prescrições: %d eventos encontrados.", len(eventos))
        return eventos

    def _comparar_prescricoes(self, medicamento: str, anterior: pd.Series, atual: pd.Series) -> list[dict]:
        eventos = []

        variacao_dose = self._variacao_relativa(anterior["dose_mg"], atual["dose_mg"])
        if abs(variacao_dose) > LIMITE_VARIACAO_DOSE:
            eventos.append({
                "data": atual["data"],
                "medicamento": medicamento,
                "tipo": "variacao_dose",
                "de": float(anterior["dose_mg"]),
                "para": float(atual["dose_mg"]),
                "variacao_percentual": round(variacao_dose * 100, 1),
                "severidade": "alta" if abs(variacao_dose) > 1.0 else "media",
            })

        variacao_frequencia = self._variacao_relativa(anterior["frequencia_horas"], atual["frequencia_horas"])
        if abs(variacao_frequencia) > LIMITE_VARIACAO_FREQUENCIA:
            eventos.append({
                "data": atual["data"],
                "medicamento": medicamento,
                "tipo": "variacao_frequencia_administracao",
                "de_horas": float(anterior["frequencia_horas"]),
                "para_horas": float(atual["frequencia_horas"]),
                "severidade": "media",
            })

        return eventos

    @staticmethod
    def _variacao_relativa(valor_anterior: float, valor_atual: float) -> float:
        if valor_anterior == 0:
            return 0.0
        return (valor_atual - valor_anterior) / valor_anterior

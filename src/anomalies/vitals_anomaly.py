"""
Detecção de anomalias em séries temporais de sinais vitais.

Combinamos duas abordagens, de propósito:

1. Regras clínicas simples (limites fixos de src/config.py) — rápidas,
   explicáveis, e são o que qualquer monitor de beira de leito já faz hoje.
2. Um classificador treinado (RandomForest, ver src/training/) — captura
   combinações incomuns entre os sinais que uma regra fixa isolada não
   pegaria (ex: FC e pressão subindo juntas de um jeito que foge do padrão).
   O modelo é treinado uma vez (python -m src.training.treinar_modelo_vitais)
   e carregado aqui pronto — nada de treinar de novo a cada execução.

Nenhuma das duas sozinha é suficiente: regras fixas perdem contexto
individual do paciente, e um modelo estatístico puro pode ignorar limites
que são perigosos clinicamente mesmo sendo "normais" estatisticamente.

Se o modelo treinado ainda não existir em disco (usuário não rodou o script
de treino), caímos de volta para um Isolation Forest ajustado na hora — mais
fraco, mas garante que o sistema nunca trava por falta de um arquivo.
"""

import logging

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import CONTAMINACAO_ISOLATION_FOREST, LIMITES_SINAIS_VITAIS
from src.training.utils import carregar_modelo_treinado

logger = logging.getLogger(__name__)

COLUNAS_SINAIS = ["frequencia_cardiaca", "spo2", "pressao_sistolica", "pressao_diastolica"]
TAMANHO_JANELA_MIN = 15  # precisa bater com src/training/conjunto_dados_vitais.py
NOME_MODELO = "vitais_rf"


def _fechar_episodio(episodio: dict) -> dict:
    """
    Transforma o episódio acumulado (início/fim/lista de valores) no formato
    final consumido pela fusão multimodal e pelo relatório. Mantemos as
    chaves "timestamp" e "valor" (usando o início e o valor de pico do
    episódio) para que o resto do sistema não precise saber que, por baixo,
    isso veio de um agrupamento de vários minutos.
    """
    valores = episodio["valores"]
    if episodio["origem_deteccao"] == "limite_clinico":
        valor_pico = max(valores, key=lambda v: abs(v))
    else:
        valor_pico = valores[-1]  # já é um dicionário {metrica: valor}, mantemos o mais recente

    return {
        "timestamp": episodio["inicio"],
        "inicio": episodio["inicio"],
        "fim": episodio["fim"],
        "metrica": episodio["metrica"],
        "valor": valor_pico,
        "duracao_minutos": round((episodio["fim"] - episodio["inicio"]).total_seconds() / 60, 1),
        "ocorrencias": episodio["ocorrencias"],
        "origem_deteccao": episodio["origem_deteccao"],
        "severidade": episodio["severidade"],
    }


class DetectorAnomaliasVitais:
    def __init__(self):
        self._modelo_treinado = carregar_modelo_treinado(NOME_MODELO)
        if self._modelo_treinado is None:
            logger.warning(
                "Modelo treinado '%s' não encontrado em models/. Usando Isolation Forest "
                "ajustado na hora (mais fraco). Rode: python -m src.training.treinar_modelo_vitais",
                NOME_MODELO,
            )
        self._modelo_fallback = IsolationForest(contamination=CONTAMINACAO_ISOLATION_FOREST, random_state=42)

    def detectar(self, sinais_vitais: pd.DataFrame) -> list[dict]:
        """
        Recebe um DataFrame com colunas timestamp + sinais vitais e devolve
        uma lista de EPISÓDIOS de anomalia (não uma linha por minuto
        anômalo). Minutos anômalos consecutivos da mesma métrica são
        agrupados num único episódio com início, fim e valor de pico — é
        assim que uma equipe médica de fato quer receber o alerta: "SpO2
        baixo entre 11:05 e 11:15", e não onze alertas separados de um
        minuto cada.
        """
        colunas_presentes = [c for c in COLUNAS_SINAIS if c in sinais_vitais.columns]
        if not colunas_presentes:
            raise ValueError(f"Nenhuma coluna de sinal vital reconhecida. Esperado alguma de: {COLUNAS_SINAIS}")

        eventos_por_minuto = []
        eventos_por_minuto += self._checar_limites_clinicos(sinais_vitais, colunas_presentes)

        if self._modelo_treinado is not None:
            eventos_por_minuto += self._checar_modelo_treinado(sinais_vitais, colunas_presentes)
        else:
            eventos_por_minuto += self._checar_estatistico_fallback(sinais_vitais, colunas_presentes)

        eventos_por_minuto.sort(key=lambda e: e["timestamp"])

        episodios = self._agrupar_em_episodios(eventos_por_minuto)
        logger.info(
            "Detecção de anomalias em sinais vitais: %d minutos anômalos agrupados em %d episódio(s).",
            len(eventos_por_minuto), len(episodios),
        )
        return episodios

    @staticmethod
    def _agrupar_em_episodios(eventos_por_minuto: list[dict], tolerancia_minutos: int = 2) -> list[dict]:
        """Agrupa eventos consecutivos da mesma métrica em episódios únicos."""
        eventos_por_metrica: dict[str, list[dict]] = {}
        for evento in eventos_por_minuto:
            eventos_por_metrica.setdefault(evento["metrica"], []).append(evento)

        episodios = []
        for metrica, eventos in eventos_por_metrica.items():
            episodio_atual = None
            for evento in eventos:
                if episodio_atual and (evento["timestamp"] - episodio_atual["fim"]) <= pd.Timedelta(minutes=tolerancia_minutos):
                    episodio_atual["fim"] = evento["timestamp"]
                    episodio_atual["ocorrencias"] += 1
                    episodio_atual["valores"].append(evento["valor"])
                else:
                    if episodio_atual:
                        episodios.append(_fechar_episodio(episodio_atual))
                    episodio_atual = {
                        "metrica": metrica, "inicio": evento["timestamp"], "fim": evento["timestamp"],
                        "ocorrencias": 1, "valores": [evento["valor"]],
                        "origem_deteccao": evento["origem_deteccao"], "severidade": evento["severidade"],
                    }
            if episodio_atual:
                episodios.append(_fechar_episodio(episodio_atual))

        # Um único minuto "estatisticamente estranho" isolado costuma ser
        # ruído de medição, não um evento clínico real — só promovemos a
        # alerta episódios de modelo/estatística que se sustentam por 2+
        # minutos consecutivos. Já os episódios de limite clínico (ex: SpO2
        # abaixo de 92%) são mantidos mesmo com uma única ocorrência, porque
        # cruzar um limiar fisiológico, mesmo que brevemente, importa.
        episodios = [
            e for e in episodios
            if e["origem_deteccao"] == "limite_clinico" or e["ocorrencias"] >= 2
        ]

        episodios.sort(key=lambda e: e["timestamp"])
        return episodios

    def _checar_limites_clinicos(self, df: pd.DataFrame, colunas: list[str]) -> list[dict]:
        eventos = []
        for coluna in colunas:
            if coluna not in LIMITES_SINAIS_VITAIS:
                continue
            limites = LIMITES_SINAIS_VITAIS[coluna]
            fora_do_limite = (df[coluna] < limites["min"]) | (df[coluna] > limites["max"])

            for _, linha in df[fora_do_limite].iterrows():
                eventos.append({
                    "timestamp": linha["timestamp"],
                    "metrica": coluna,
                    "valor": round(float(linha[coluna]), 1),
                    "origem_deteccao": "limite_clinico",
                    "severidade": "alta",
                })
        return eventos

    def _checar_modelo_treinado(self, df: pd.DataFrame, colunas: list[str]) -> list[dict]:
        """
        Aplica o RandomForest treinado (src/training/treinar_modelo_vitais.py)
        sobre janelas deslizantes de `TAMANHO_JANELA_MIN` minutos: a cada
        minuto, olha para os últimos N minutos, resume em estatísticas
        (média/desvio/mín/máx — os MESMOS atributos usados no treino) e
        pergunta ao modelo se aquela janela parece anômala.
        """
        atributos = {}
        for coluna in colunas:
            janela = df[coluna].rolling(TAMANHO_JANELA_MIN)
            atributos[f"{coluna}_media"] = janela.mean()
            atributos[f"{coluna}_desvio"] = janela.std()
            atributos[f"{coluna}_min"] = janela.min()
            atributos[f"{coluna}_max"] = janela.max()

        X = pd.DataFrame(atributos)
        indices_validos = X.dropna().index  # primeiros (janela-1) minutos não têm janela completa

        if len(indices_validos) == 0:
            return []

        X_valido = X.loc[indices_validos]
        previsoes = self._modelo_treinado.predict(X_valido)
        probabilidades = self._modelo_treinado.predict_proba(X_valido)[:, 1]

        eventos = []
        for indice, previsto, probabilidade in zip(indices_validos, previsoes, probabilidades):
            if previsto != 1:
                continue
            linha = df.loc[indice]
            eventos.append({
                "timestamp": linha["timestamp"],
                "metrica": "modelo_treinado_vitais",
                "valor": {c: round(float(linha[c]), 1) for c in colunas},
                "origem_deteccao": "modelo_treinado",
                "severidade": "alta" if probabilidade >= 0.8 else "media",
            })
        return eventos

    def _checar_estatistico_fallback(self, df: pd.DataFrame, colunas: list[str]) -> list[dict]:
        """Isolation Forest ajustado na hora — usado só se não houver modelo treinado salvo."""
        matriz = df[colunas].to_numpy()
        previsao = self._modelo_fallback.fit_predict(matriz)  # -1 = anômalo, 1 = normal

        eventos = []
        for indice, classificacao in enumerate(previsao):
            if classificacao != -1:
                continue
            linha = df.iloc[indice]
            eventos.append({
                "timestamp": linha["timestamp"],
                "metrica": "combinacao_sinais_vitais",
                "valor": {c: round(float(linha[c]), 1) for c in colunas},
                "origem_deteccao": "padrao_estatistico",
                "severidade": "media",
            })
        return eventos

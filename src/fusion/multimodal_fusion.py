"""
Fusão multimodal: junta os achados de vídeo, áudio, sinais vitais e
prescrições numa única pontuação de risco por paciente, e dispara os
alertas correspondentes.

A fusão aqui é "tardia" (late fusion): cada modalidade é processada e
analisada separadamente por seu próprio especialista (visão, áudio,
séries temporais), e só no final combinamos os resultados — não os
atributos brutos. Essa abordagem é mais simples de explicar e de manter do
que uma fusão "precoce" (concatenar atributos de fontes tão diferentes num
único modelo), e é a mais comum em sistemas clínicos multimodais reais.
"""

import logging
from collections import Counter

from src.anomalies.alert_manager import GerenciadorDeAlertas

logger = logging.getLogger(__name__)

PESOS_RISCO = {
    "vitais_limite_clinico": 3,
    "vitais_modelo_treinado": 2,
    "vitais_padrao_estatistico": 2,  # usado só quando o modelo treinado não está disponível (fallback)
    "prescricao": 2,
    "movimento": 2,
    "audio_fadiga": 1,
    "audio_alteracao_fala": 1,
    "audio_termo_critico": 3,
    "audio_evento_sonoro_grave": 3,
}

# Eventos do AudioSet (via YAMNet) que merecem atenção imediata quando
# aparecem entre os sons mais prováveis do áudio — diferente de sons comuns
# de uma consulta (fala, respiração, tosse leve), que não elevam o risco.
EVENTOS_SONOROS_GRAVES = {"Vomiting", "Choking", "Screaming", "Groan", "Baby cry, infant cry"}
ROTULOS_SENTIMENTO = {
    "positive": "positivo",
    "negative": "negativo",
    "neutral": "neutro",
    "mixed": "misto",
    "indefinido": "indefinido",
}


def calcular_risco_paciente(
    paciente_id: str,
    gerenciador_alertas: GerenciadorDeAlertas,
    anomalias_vitais: list[dict] | None = None,
    anomalias_prescricao: list[dict] | None = None,
    resultado_video: dict | None = None,
    resultado_audio: dict | None = None,
) -> dict:
    """
    Recebe os resultados já calculados de cada modalidade (podem vir
    parciais/ausentes — nem todo paciente terá vídeo ou áudio no mesmo dia)
    e devolve uma pontuação de risco consolidada, junto com os alertas já
    registrados no GerenciadorDeAlertas.
    """
    pontuacao = 0
    motivos = []

    for anomalia in anomalias_vitais or []:
        origem = f"vitais_{anomalia['origem_deteccao']}"
        pontuacao += PESOS_RISCO.get(origem, 1)
        duracao = anomalia.get("duracao_minutos", 0)
        janela = f"{anomalia['inicio']} a {anomalia['fim']}" if duracao else str(anomalia["timestamp"])
        mensagem = f"Episódio de {anomalia['metrica']} fora do padrão entre {janela} ({duracao} min, {anomalia.get('ocorrencias', 1)} ocorrência(s))"
        motivos.append(mensagem)
        gerenciador_alertas.registrar(paciente_id, "vitais", mensagem, severidade=anomalia["severidade"])

    for anomalia in anomalias_prescricao or []:
        pontuacao += PESOS_RISCO["prescricao"]
        mensagem = f"Alteração incomum na prescrição de {anomalia['medicamento']} ({anomalia['tipo']})"
        motivos.append(mensagem)
        gerenciador_alertas.registrar(paciente_id, "prescricao", mensagem, severidade=anomalia["severidade"])

    if resultado_video and resultado_video.get("possivel_desvio_no_procedimento"):
        anomalias_movimento = resultado_video.get("anomalias_movimento") or []
        n_eventos = len(anomalias_movimento)
        pontuacao += PESOS_RISCO["movimento"] * n_eventos

        contagem_motivos = Counter()
        severidades = []
        for evento in anomalias_movimento:
            if not isinstance(evento, dict):
                continue
            severidade_evento = evento.get("severidade")
            if severidade_evento:
                severidades.append(severidade_evento)
            for motivo in (evento.get("motivos") or []):
                contagem_motivos[str(motivo)] += 1

        principais_motivos = [m for m, _c in contagem_motivos.most_common(3)]
        if principais_motivos:
            mensagem = (
                f"{n_eventos} evento(s) de movimento fora do padrão detectado(s) em vídeo. "
                f"Principais motivos: {', '.join(principais_motivos)}"
            )
        else:
            mensagem = f"{n_eventos} evento(s) de movimento fora do padrão detectado(s) em vídeo"

        motivos.append(mensagem)

        severidade_alerta = "media"
        if "alta" in severidades:
            severidade_alerta = "alta"
        gerenciador_alertas.registrar(paciente_id, "movimento", mensagem, severidade=severidade_alerta)

    if resultado_audio:
        classificacao_modelo = resultado_audio.get("classificacao_modelo_treinado")
        indicadores = resultado_audio.get("indicadores_acusticos", {})

        if classificacao_modelo is not None:
            # Caminho preferido: modelo treinado (src/training/treinar_modelo_audio.py).
            if classificacao_modelo["classe"] == "alterado":
                pontuacao += PESOS_RISCO["audio_alteracao_fala"]
                mensagem = (
                    f"Modelo treinado indicou alteração na voz do paciente "
                    f"(confiança {classificacao_modelo['confianca'] * 100:.0f}%)"
                )
                motivos.append(mensagem)
                gerenciador_alertas.registrar(paciente_id, "audio", mensagem, severidade="media")
        else:
            # Fallback: indicadores heurísticos (usados só se o modelo não foi treinado ainda).
            if indicadores.get("indicador_fadiga", 0) > 0.6:
                pontuacao += PESOS_RISCO["audio_fadiga"]
                mensagem = f"Indício de fadiga na voz do paciente (pontuação {indicadores['indicador_fadiga']})"
                motivos.append(mensagem)
                gerenciador_alertas.registrar(paciente_id, "audio", mensagem, severidade="baixa")

            if indicadores.get("indicador_alteracao_fala", 0) > 0.6:
                pontuacao += PESOS_RISCO["audio_alteracao_fala"]
                mensagem = f"Indício de alteração de fala / possível disartria (pontuação {indicadores['indicador_alteracao_fala']})"
                motivos.append(mensagem)
                gerenciador_alertas.registrar(paciente_id, "audio", mensagem, severidade="media")

        analise_texto = resultado_audio.get("analise_texto") or {}
        sentimento = analise_texto.get("sentimento")
        if sentimento:
            rotulo_sentimento = ROTULOS_SENTIMENTO.get(sentimento.lower(), sentimento)
            motivos.append(f"Sentimento geral identificado na transcrição: {rotulo_sentimento}")

        if analise_texto.get("termos_criticos_encontrados"):
            pontuacao += PESOS_RISCO["audio_termo_critico"] * len(analise_texto["termos_criticos_encontrados"])
            mensagem = f"Termo(s) crítico(s) mencionado(s) na consulta: {', '.join(analise_texto['termos_criticos_encontrados'])}"
            motivos.append(mensagem)
            gerenciador_alertas.registrar(paciente_id, "audio", mensagem, severidade="alta")

        eventos_sonoros = resultado_audio.get("eventos_sonoros_audioset") or []
        eventos_graves = [e for e in eventos_sonoros if e.get("classe") in EVENTOS_SONOROS_GRAVES]
        if eventos_graves:
            pontuacao += PESOS_RISCO["audio_evento_sonoro_grave"] * len(eventos_graves)
            nomes = ", ".join(f"{e['classe']} ({e['pontuacao'] * 100:.0f}%)" for e in eventos_graves)
            mensagem = f"YAMNet (AudioSet) identificou som(ns) de atenção no áudio: {nomes}"
            motivos.append(mensagem)
            gerenciador_alertas.registrar(paciente_id, "audio", mensagem, severidade="alta")

    nivel_risco = _classificar_risco(pontuacao)
    logger.info("Risco consolidado do paciente %s: %d pontos (%s)", paciente_id, pontuacao, nivel_risco)

    return {
        "paciente_id": paciente_id,
        "pontuacao_risco": pontuacao,
        "nivel_risco": nivel_risco,
        "motivos": motivos,
    }


def _classificar_risco(pontuacao: int) -> str:
    if pontuacao >= 8:
        return "alto"
    if pontuacao >= 3:
        return "medio"
    return "baixo"

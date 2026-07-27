"""
Geração de um parecer clínico final com API da OpenAI a partir dos resultados
multimodais (sinais vitais, vídeo e áudio).

Saída principal:
- pontuacao_paciente_0a10: 0 = estado muito ruim, 10 = muito saudável
- item_maior_risco: modalidade ou fator mais crítico
- alerta_equipe_medica: orientação textual objetiva para priorização
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from src import config

logger = logging.getLogger(__name__)


PROMPT_SISTEMA = (
    "Você é um assistente clínico de apoio à triagem hospitalar. "
    "Seu papel é gerar um resumo objetivo para equipe médica com base em análises "
    "de sinais vitais, vídeo e áudio. "
    "Nunca invente exames, medicamentos ou diagnósticos confirmatórios. "
    "Seja conservador e explícito sobre risco."
)


def _extrair_json_de_texto(texto: str) -> dict[str, Any]:
    """
    Tenta interpretar JSON mesmo quando o modelo responde com texto extra
    ou bloco markdown (```json ... ```).
    """
    conteudo = texto.strip()
    if not conteudo:
        raise json.JSONDecodeError("resposta vazia", conteudo, 0)

    # Caminho feliz: resposta já veio como JSON puro.
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError:
        pass

    # Tenta extrair de bloco markdown.
    if "```" in conteudo:
        partes = conteudo.split("```")
        for parte in partes:
            candidato = parte.strip()
            if not candidato:
                continue
            if candidato.lower().startswith("json"):
                candidato = candidato[4:].strip()
            if not candidato:
                continue
            try:
                return json.loads(candidato)
            except json.JSONDecodeError:
                continue

    # Último fallback: pega do primeiro "{" até o último "}".
    inicio = conteudo.find("{")
    fim = conteudo.rfind("}")
    if inicio != -1 and fim != -1 and fim > inicio:
        return json.loads(conteudo[inicio:fim + 1])

    raise json.JSONDecodeError("json nao encontrado", conteudo, 0)


def _extrair_resumo_para_prompt(detalhes: dict[str, Any], resultado_risco: dict[str, Any]) -> dict[str, Any]:
    """Reduz o payload para os campos relevantes do LLM, evitando enviar ruído."""
    vitais = detalhes.get("vitais") or []
    video = detalhes.get("video") or {}
    audio = detalhes.get("audio") or {}

    # Resumo de anomalias de sinais vitais (evita mandar listas extensas)
    total_vitais = len(vitais) if isinstance(vitais, list) else 0
    severidade_vitais = Counter()
    metricas_vitais = Counter()
    exemplos_vitais = []
    if isinstance(vitais, list):
        for item in vitais:
            if not isinstance(item, dict):
                continue
            severidade_vitais[str(item.get("severidade", "desconhecida"))] += 1
            metricas_vitais[str(item.get("metrica", "desconhecida"))] += 1
        for item in vitais[:5]:
            if isinstance(item, dict):
                exemplos_vitais.append(
                    {
                        "metrica": item.get("metrica"),
                        "severidade": item.get("severidade"),
                        "origem": item.get("origem_deteccao"),
                        "duracao_minutos": item.get("duracao_minutos"),
                    }
                )

    # Resumo de vídeo
    anomalias_movimento = video.get("anomalias_movimento") or []
    total_anomalias_video = len(anomalias_movimento) if isinstance(anomalias_movimento, list) else 0
    motivos_video = Counter()
    if isinstance(anomalias_movimento, list):
        for evento in anomalias_movimento[:50]:
            if not isinstance(evento, dict):
                continue
            for motivo in (evento.get("motivos") or []):
                motivos_video[str(motivo)] += 1
    top_motivos_video = [{"motivo": m, "ocorrencias": c} for m, c in motivos_video.most_common(5)]

    # Resumo de áudio
    eventos_audio = audio.get("eventos_sonoros_audioset") or []
    eventos_audio_top = []
    if isinstance(eventos_audio, list):
        for item in eventos_audio[:5]:
            if isinstance(item, dict):
                eventos_audio_top.append(
                    {
                        "classe": item.get("classe"),
                        "pontuacao": item.get("pontuacao"),
                        "relevante_clinicamente": item.get("relevante_clinicamente"),
                    }
                )

    motivos_risco = resultado_risco.get("motivos", []) or []

    resumo = {
        "risco_atual_motor_regra": {
            "nivel": resultado_risco.get("nivel_risco"),
            "pontuacao": resultado_risco.get("pontuacao_risco"),
            "total_motivos": len(motivos_risco),
            "motivos_top8": motivos_risco[:8],
        },
        "sinais_vitais": {
            "total_anomalias": total_vitais,
            "contagem_por_severidade": dict(severidade_vitais),
            "contagem_por_metrica": dict(metricas_vitais),
            "exemplos": exemplos_vitais,
        },
        "video": {
            "possivel_desvio_no_procedimento": video.get("possivel_desvio_no_procedimento"),
            "total_frames_analisados": video.get("total_frames_analisados"),
            "total_anomalias_movimento": total_anomalias_video,
            "top_motivos_anomalia": top_motivos_video,
            "objetos_detectados": video.get("objetos_detectados"),
        }
        if video
        else None,
        "audio": {
            "classificacao_modelo_treinado": audio.get("classificacao_modelo_treinado"),
            "indicadores_acusticos": audio.get("indicadores_acusticos"),
            "analise_texto": audio.get("analise_texto"),
            "eventos_sonoros_audioset_top5": eventos_audio_top,
            "avisos": audio.get("avisos"),
        }
        if audio
        else None,
    }
    return resumo


def _resumo_minimo_para_fallback(resumo: dict[str, Any]) -> dict[str, Any]:
    """Resumo ultracompacto para retry quando há rate limit por tokens."""
    risco = resumo.get("risco_atual_motor_regra") or {}
    vitais = resumo.get("sinais_vitais") or {}
    video = resumo.get("video") or {}
    audio = resumo.get("audio") or {}

    return {
        "risco_atual_motor_regra": {
            "nivel": risco.get("nivel"),
            "pontuacao": risco.get("pontuacao"),
            "motivos_top3": (risco.get("motivos") or [])[:3],
        },
        "sinais_vitais": {
            "total_anomalias": vitais.get("total_anomalias"),
            "contagem_por_severidade": vitais.get("contagem_por_severidade"),
            "contagem_por_metrica": vitais.get("contagem_por_metrica"),
        },
        "video": {
            "desvio": video.get("possivel_desvio_no_procedimento"),
            "total_anomalias_movimento": video.get("total_anomalias_movimento"),
        },
        "audio": {
            "classe_modelo": (audio.get("classificacao_modelo_treinado") or {}).get("classe"),
            "termos_criticos": (audio.get("analise_texto") or {}).get("termos_criticos_encontrados"),
        },
    }


def _chamar_openai(cliente: Any, prompt_usuario: dict[str, Any], model: str) -> str:
    """Isola chamada ao modelo para facilitar retry com payload reduzido."""
    resposta = cliente.responses.create(
        model=model,
        input=[
            {"role": "system", "content": PROMPT_SISTEMA},
            # Alguns resultados podem trazer tipos não nativos (ex.: numpy/pandas);
            # `default=str` evita quebra de serialização no envio para o modelo.
            {"role": "user", "content": json.dumps(prompt_usuario, ensure_ascii=False, default=str)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "parecer_clinico_final",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pontuacao_paciente_0a10": {"type": "integer", "minimum": 0, "maximum": 10},
                        "item_maior_risco": {"type": "string"},
                        "alerta_equipe_medica": {"type": "string"},
                        "justificativa_curta": {"type": "string"},
                    },
                    "required": [
                        "pontuacao_paciente_0a10",
                        "item_maior_risco",
                        "alerta_equipe_medica",
                        "justificativa_curta",
                    ],
                },
            }
        },
        temperature=0.1,
        max_output_tokens=220,
    )
    return (resposta.output_text or "").strip()


def _normalizar_pontuacao_0a10(valor: Any) -> int:
    """Garante inteiro entre 0 e 10 para manter consistência no relatório."""
    try:
        numero = int(round(float(valor)))
    except (TypeError, ValueError):
        return 5
    return max(0, min(10, numero))


def gerar_parecer_openai(
    paciente_id: str,
    resultado_risco: dict[str, Any],
    detalhes: dict[str, Any],
) -> dict[str, Any]:
    """
    Usa a API da OpenAI para gerar parecer final com score 0-10 de saúde,
    item de maior risco e alerta para equipe médica.
    """
    if not config.openai_configurado():
        raise RuntimeError(
            "OpenAI não configurada. Defina OPENAI_API_KEY no .env para gerar o relatório final com LLM."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as erro:
        raise RuntimeError(
            "Pacote 'openai' não instalado no ambiente atual. Rode: python -m pip install openai"
        ) from erro

    cliente = OpenAI(api_key=config.OPENAI_API_KEY)
    resumo = _extrair_resumo_para_prompt(detalhes, resultado_risco)

    prompt_usuario = {
        "paciente_id": paciente_id,
        "instrucao": (
            "Com base apenas nos dados fornecidos, gere JSON estrito com os campos: "
            "pontuacao_paciente_0a10 (inteiro 0-10, onde 0 = muito ruim e 10 = muito saudável), "
            "item_maior_risco (string curta), "
            "alerta_equipe_medica (mensagem objetiva, 1-3 frases), "
            "justificativa_curta (string curta)."
        ),
        "dados": resumo,
    }

    modelo_usado = config.OPENAI_MODEL
    try:
        texto = _chamar_openai(cliente, prompt_usuario, modelo_usado)
    except Exception as erro:
        # Retry com payload mínimo em caso de erro de tokens por minuto.
        mensagem = str(erro).lower()
        if "request too large" in mensagem or "tokens per min" in mensagem or "rate_limit_exceeded" in mensagem:
            logger.warning("Rate limit por tokens; tentando novamente com payload reduzido.")
            prompt_usuario["dados"] = _resumo_minimo_para_fallback(resumo)
            try:
                texto = _chamar_openai(cliente, prompt_usuario, modelo_usado)
            except Exception as erro_fallback:
                mensagem_fallback = str(erro_fallback).lower()
                # Se ainda houver rate limit, tenta com modelo de fallback (mais econômico).
                if (
                    config.OPENAI_FALLBACK_MODEL
                    and config.OPENAI_FALLBACK_MODEL != modelo_usado
                    and (
                        "request too large" in mensagem_fallback
                        or "tokens per min" in mensagem_fallback
                        or "rate_limit_exceeded" in mensagem_fallback
                    )
                ):
                    modelo_usado = config.OPENAI_FALLBACK_MODEL
                    logger.warning(
                        "Rate limit persistente; tentando modelo de fallback: %s",
                        modelo_usado,
                    )
                    texto = _chamar_openai(cliente, prompt_usuario, modelo_usado)
                else:
                    raise
        else:
            raise

    if not texto:
        raise RuntimeError("OpenAI retornou resposta vazia ao gerar parecer clínico.")

    try:
        estrutura = _extrair_json_de_texto(texto)
    except json.JSONDecodeError as erro:
        logger.warning("Falha ao interpretar JSON do parecer OpenAI: %s", erro)
        raise RuntimeError(
            "Não foi possível interpretar a resposta da OpenAI em JSON. Ajuste o prompt/modelo e tente novamente."
        ) from erro

    parecer = {
        "pontuacao_paciente_0a10": _normalizar_pontuacao_0a10(estrutura.get("pontuacao_paciente_0a10")),
        "item_maior_risco": str(estrutura.get("item_maior_risco", "não identificado")),
        "alerta_equipe_medica": str(estrutura.get("alerta_equipe_medica", "")),
        "justificativa_curta": str(estrutura.get("justificativa_curta", "")),
        "modelo_openai": modelo_usado,
    }
    return parecer

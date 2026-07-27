"""
Geração do relatório automático final de um paciente, juntando os
resultados de todas as modalidades e o score de risco consolidado.

O relatório é salvo tanto em JSON (para integração com outros sistemas)
quanto em Markdown (para leitura humana pela equipe médica) — é o
"gerar relatórios automáticos indicando desvios ou falhas" pedido no
desafio, agora unificado para todas as fontes de dado, não só vídeo.
"""

import json
import logging
from datetime import datetime

from src.config import PASTA_RELATORIOS

logger = logging.getLogger(__name__)


def gerar_relatorio(paciente_id: str, resultado_risco: dict, detalhes: dict) -> dict:
    """
    Monta o relatório final, salva em disco (JSON + Markdown) e devolve o
    conteúdo também em memória, para quem chamou decidir o que fazer com ele
    (ex: mostrar num painel, enviar por e-mail).
    """
    momento = datetime.now()
    relatorio = {
        "paciente_id": paciente_id,
        "gerado_em": momento.isoformat(),
        "nivel_risco": resultado_risco["nivel_risco"],
        "pontuacao_risco": resultado_risco["pontuacao_risco"],
        "motivos": resultado_risco["motivos"],
        "resumo_openai": resultado_risco.get("resumo_openai"),
        "detalhes_por_modalidade": detalhes,
    }

    prefixo = f"{paciente_id}_{momento.strftime('%Y%m%d_%H%M%S')}"
    caminho_json = PASTA_RELATORIOS / f"{prefixo}.json"
    caminho_md = PASTA_RELATORIOS / f"{prefixo}.md"

    caminho_json.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    caminho_md.write_text(_para_markdown(relatorio), encoding="utf-8")

    logger.info("Relatório gerado: %s", caminho_md)
    return relatorio


def _para_markdown(relatorio: dict) -> str:
    linhas = [
        f"# Relatório de monitoramento — Paciente {relatorio['paciente_id']}",
        "",
        f"**Gerado em:** {relatorio['gerado_em']}",
        f"**Nível de risco:** {relatorio['nivel_risco'].upper()} (pontuação: {relatorio['pontuacao_risco']})",
        "",
        "## Motivos identificados",
    ]
    if relatorio["motivos"]:
        linhas += [f"- {motivo}" for motivo in relatorio["motivos"]]
    else:
        linhas.append("- Nenhuma anomalia relevante identificada no período analisado.")

    resumo_openai = relatorio.get("resumo_openai")
    if resumo_openai:
        linhas += [
            "",
            "## Parecer final (OpenAI)",
            f"- **Pontuação do paciente (0-10):** {resumo_openai.get('pontuacao_paciente_0a10', 'n/d')}",
            f"- **Item de maior risco:** {resumo_openai.get('item_maior_risco', 'n/d')}",
            f"- **Alerta para equipe médica:** {resumo_openai.get('alerta_equipe_medica', 'n/d')}",
            f"- **Justificativa curta:** {resumo_openai.get('justificativa_curta', 'n/d')}",
            f"- **Modelo:** {resumo_openai.get('modelo_openai', 'n/d')}",
        ]

    linhas += ["", "## Detalhes por modalidade", "```json", json.dumps(relatorio["detalhes_por_modalidade"], indent=2, ensure_ascii=False, default=str), "```"]
    return "\n".join(linhas)

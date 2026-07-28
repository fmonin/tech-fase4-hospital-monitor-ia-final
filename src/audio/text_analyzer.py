"""
Análise de texto da transcrição usando o Azure Text Analytics: sentimento
geral da fala do paciente e identificação de termos clinicamente relevantes.
"""

import logging

from src import config

logger = logging.getLogger(__name__)

# Termos que, se aparecerem na fala do paciente, merecem atenção imediata da
# equipe médica. Lista pequena e editável — em produção viria de um
# vocabulário clínico validado, não de uma lista fixa como esta.
TERMOS_CRITICOS = [
    "dor no peito", "falta de ar", "tontura", "desmaio", "sangramento",
    "não consigo respirar", "dor muito forte", "formigamento",
]


def analisar_texto(transcricao: str, idioma: str = "pt") -> dict:
    """
    Devolve sentimento (positivo/neutro/negativo), frases-chave e termos
    críticos encontrados na transcrição de uma consulta.
    """
    if not transcricao.strip():
        return {"sentimento": "indefinido", "frases_chave": [], "termos_criticos_encontrados": []}

    termos_encontrados = [termo for termo in TERMOS_CRITICOS if termo in transcricao.lower()]

    if not config.azure_text_analytics_configurado():
        raise RuntimeError(
            "Azure Text Analytics não configurado. Defina AZURE_TEXT_ANALYTICS_KEY e "
            "AZURE_TEXT_ANALYTICS_ENDPOINT no arquivo .env antes de analisar o texto."
        )

    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cliente = TextAnalyticsClient(
        endpoint=config.AZURE_TEXT_ANALYTICS_ENDPOINT,
        credential=AzureKeyCredential(config.AZURE_TEXT_ANALYTICS_KEY),
    )

    documentos = [transcricao]
    resultado_sentimento = cliente.analyze_sentiment(documentos, language=idioma)[0]
    resultado_frases = cliente.extract_key_phrases(documentos, language=idioma)[0]

    logger.info("Análise de texto concluída: sentimento=%s", resultado_sentimento.sentiment)

    return {
        "sentimento": resultado_sentimento.sentiment,
        "confianca_sentimento": {
            "positivo": resultado_sentimento.confidence_scores.positive,
            "neutro": resultado_sentimento.confidence_scores.neutral,
            "negativo": resultado_sentimento.confidence_scores.negative,
        },
        "trechos_sentimento": [
            {
                "trecho": sentenca.text,
                "sentimento": sentenca.sentiment,
                "confianca": {
                    "positivo": sentenca.confidence_scores.positive,
                    "neutro": sentenca.confidence_scores.neutral,
                    "negativo": sentenca.confidence_scores.negative,
                },
            }
            for sentenca in (resultado_sentimento.sentences or [])
            if getattr(sentenca, "text", "").strip()
        ],
        "frases_chave": list(resultado_frases.key_phrases),
        "termos_criticos_encontrados": termos_encontrados,
    }

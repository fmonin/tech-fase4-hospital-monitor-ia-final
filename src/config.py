"""
Configurações centrais do projeto.

A ideia aqui é simples: todo mundo que mexe em uma credencial ou em um
limiar clínico (threshold) mexe neste arquivo, e só neste arquivo.
Isso evita "números mágicos" espalhados pelo código.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# --- Caminhos do projeto ---
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_AMOSTRAS = RAIZ_PROJETO / "amostras"
PASTA_RELATORIOS = RAIZ_PROJETO / "reports"
PASTA_RELATORIOS.mkdir(exist_ok=True)

# Carrega variáveis do arquivo .env da raiz do projeto, se existir (uso local).
# Em produção, as variáveis normalmente já vêm do ambiente (ex: App Service, Docker).
load_dotenv(dotenv_path=RAIZ_PROJETO / ".env")

# --- Credenciais Azure (ficam vazias até o usuário configurar o .env) ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_TEXT_ANALYTICS_KEY = os.getenv("AZURE_TEXT_ANALYTICS_KEY", "")
AZURE_TEXT_ANALYTICS_ENDPOINT = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT", "")

# --- OpenAI (relatório final assistido por LLM) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

# --- Notificação externa opcional de alertas (webhook HTTP) ---
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# --- Backend de pose para análise de vídeo ---
# auto: tenta OpenPose primeiro, cai para MediaPipe se indisponível.
# openpose: exige OpenPose configurado (erro se não estiver).
# mediapipe: força MediaPipe.
POSE_BACKEND = os.getenv("POSE_BACKEND", "auto").strip().lower()
OPENPOSE_MODEL_FOLDER = os.getenv("OPENPOSE_MODEL_FOLDER", "")
MEDIAPIPE_POSETASK_MODEL_PATH = os.getenv("MEDIAPIPE_POSETASK_MODEL_PATH", "")

# --- Áreas críticas (normalizadas, 0-1) para monitoramento em vídeo ---
# Ex.: mesa cirúrgica, zona estéril, área de equipamento sensível.
AREAS_CRITICAS_VIDEO = [
    {"nome": "area_central_procedimento", "xmin": 0.30, "xmax": 0.70, "ymin": 0.25, "ymax": 0.80},
    {"nome": "area_equipamentos", "xmin": 0.75, "xmax": 1.00, "ymin": 0.10, "ymax": 0.90},
]


def azure_speech_configurado() -> bool:
    """Confirma se há credenciais suficientes para chamar o Azure Speech."""
    return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)


def azure_text_analytics_configurado() -> bool:
    """Confirma se há credenciais suficientes para chamar o Azure Text Analytics."""
    return bool(AZURE_TEXT_ANALYTICS_KEY and AZURE_TEXT_ANALYTICS_ENDPOINT)


def openai_configurado() -> bool:
    """Confirma se há credencial para chamada da API da OpenAI."""
    return bool(OPENAI_API_KEY)


def alerta_webhook_configurado() -> bool:
    """Confirma se há URL de webhook para envio de alertas automáticos."""
    return bool(ALERT_WEBHOOK_URL)


# --- Limiares clínicos usados na detecção de anomalias de sinais vitais ---
# Faixas de referência aproximadas para um adulto em repouso. Não substituem
# critério médico: servem como ponto de partida para os alertas automáticos.
LIMITES_SINAIS_VITAIS = {
    "frequencia_cardiaca": {"min": 50, "max": 120},   # bpm
    "spo2": {"min": 92, "max": 100},                  # % saturação de O2
    "pressao_sistolica": {"min": 90, "max": 140},      # mmHg
    "pressao_diastolica": {"min": 60, "max": 90},      # mmHg
}

# Sensibilidade do modelo estatístico (Isolation Forest) de anomalias.
# Quanto menor, mais o modelo tolera variações antes de marcar como anômalo.
CONTAMINACAO_ISOLATION_FOREST = 0.05

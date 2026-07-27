"""
Ponto único de saída de alertas para a equipe médica.

Todos os detectores (vitais, prescrição, movimento, e a fusão multimodal)
convergem para cá. Hoje o "envio" é só um log estruturado + lista em
memória — de propósito: a integração final (SMS, e-mail, painel do
hospital, pager) é uma decisão de infraestrutura de cada instituição, não
algo que deveria estar hard-coded no núcleo de detecção. Trocar o método de
envio é questão de implementar um novo `_notificar` sem mexer no resto.
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from urllib import request
from urllib.error import URLError, HTTPError

from src import config

logger = logging.getLogger(__name__)

ORDEM_SEVERIDADE = {"baixa": 0, "media": 1, "alta": 2}


@dataclass
class Alerta:
    paciente_id: str
    origem: str          # "vitais", "prescricao", "movimento", "audio"
    mensagem: str
    severidade: str       # "baixa" | "media" | "alta"
    criado_em: datetime = field(default_factory=datetime.now)


class GerenciadorDeAlertas:
    def __init__(self):
        self._alertas: list[Alerta] = []

    def registrar(self, paciente_id: str, origem: str, mensagem: str, severidade: str = "media") -> Alerta:
        alerta = Alerta(paciente_id=paciente_id, origem=origem, mensagem=mensagem, severidade=severidade)
        self._alertas.append(alerta)
        self._notificar(alerta)
        return alerta

    def alertas_do_paciente(self, paciente_id: str) -> list[Alerta]:
        return [a for a in self._alertas if a.paciente_id == paciente_id]

    def resumo_por_severidade(self) -> dict:
        resumo = {"baixa": 0, "media": 0, "alta": 0}
        for alerta in self._alertas:
            resumo[alerta.severidade] += 1
        return resumo

    def mais_criticos_primeiro(self) -> list[Alerta]:
        return sorted(self._alertas, key=lambda a: ORDEM_SEVERIDADE[a.severidade], reverse=True)

    def _notificar(self, alerta: Alerta) -> None:
        nivel_log = {"alta": logging.WARNING, "media": logging.INFO, "baixa": logging.INFO}[alerta.severidade]
        logger.log(
            nivel_log,
            "[ALERTA %s] paciente=%s origem=%s: %s",
            alerta.severidade.upper(), alerta.paciente_id, alerta.origem, alerta.mensagem,
        )
        self._notificar_webhook(alerta)

    @staticmethod
    def _notificar_webhook(alerta: Alerta) -> None:
        """Envio opcional para integração externa com equipe médica (Webhook HTTP)."""
        if not config.alerta_webhook_configurado():
            return

        payload = {
            "paciente_id": alerta.paciente_id,
            "origem": alerta.origem,
            "mensagem": alerta.mensagem,
            "severidade": alerta.severidade,
            "criado_em": alerta.criado_em.isoformat(),
        }

        corpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            config.ALERT_WEBHOOK_URL,
            data=corpo,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=5) as resposta:
                if resposta.status >= 300:
                    logger.warning("Webhook de alerta respondeu com status %s", resposta.status)
        except (HTTPError, URLError, TimeoutError) as erro:
            logger.warning("Falha ao enviar alerta para webhook: %s", erro)

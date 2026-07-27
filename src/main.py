"""
Ponto de entrada do sistema: roda o pipeline multimodal completo para um
paciente e imprime/salva o relatório final.

Modo demo (padrão, `--demo`): usa dados sintéticos e/ou reais do PhysioNet
para sinais vitais, prescrições e movimento — funciona offline, sem chaves
Azure, então qualquer pessoa consegue rodar e ver o sistema de ponta a
ponta. Vídeo e áudio reais podem ser plugados via --video e --audio.

Exemplos:
    python -m src.main --demo
    python -m src.main --demo --video amostras/videos_normais/normal_1.mp4
    python -m src.main --demo --audio amostras/audio_por_classe/normal/normal_1.wav
"""

import argparse
import logging

from src.anomalies.alert_manager import GerenciadorDeAlertas
from src.anomalies.prescription_anomaly import DetectorAnomaliaPrescricao
from src.anomalies.vitals_anomaly import DetectorAnomaliasVitais
from src.dados import carregador_physionet, carregador_vitaldb, dados_sinteticos
from src.dados.carregador_csv import carregar_sinais_vitais_de_csv
from src.fusion.multimodal_fusion import calcular_risco_paciente
from src.reports.report_generator import gerar_relatorio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def carregar_sinais_vitais(usar_dados_reais: bool, caminho_csv: str | None = None):
    """
    Prioridade de fontes: CSV explícito (--csv) > VitalDB (caso real de
    cirurgia) > BIDMC/PhysioNet (numerics de UTI) > dados sintéticos. As
    duas fontes do meio exigem internet; se falharem (ex: rede restrita),
    caímos para sintético sem quebrar o pipeline.
    """
    if caminho_csv:
        return carregar_sinais_vitais_de_csv(caminho_csv)

    if usar_dados_reais:
        try:
            return carregador_vitaldb.carregar_caso_real()
        except ConnectionError as erro:
            logger.warning("Não foi possível baixar dados reais do VitalDB (%s). Tentando PhysioNet/BIDMC...", erro)
        try:
            return carregador_physionet.carregar_sinais_vitais_reais()
        except ConnectionError as erro:
            logger.warning("Não foi possível baixar dados reais do PhysioNet (%s). Usando dados sintéticos.", erro)
    return dados_sinteticos.gerar_sinais_vitais_sinteticos()


def executar_pipeline(
    paciente_id: str,
    caminho_video: str | None,
    caminho_audio: str | None,
    usar_dados_reais: bool,
    caminho_csv: str | None = None,
):
    gerenciador_alertas = GerenciadorDeAlertas()
    detalhes = {}

    # --- Sinais vitais (CSV > VitalDB/PhysioNet reais > fallback sintético) ---
    sinais_vitais = carregar_sinais_vitais(usar_dados_reais, caminho_csv)
    anomalias_vitais = DetectorAnomaliasVitais().detectar(sinais_vitais)
    detalhes["vitais"] = {"total_amostras": len(sinais_vitais), "anomalias_encontradas": len(anomalias_vitais)}

    # --- Prescrições (sintético: não há base pública de prescrições) ---
    historico_prescricoes = dados_sinteticos.gerar_historico_prescricoes_sintetico()
    anomalias_prescricao = DetectorAnomaliaPrescricao().detectar(historico_prescricoes)
    detalhes["prescricoes"] = {"total_registros": len(historico_prescricoes), "anomalias_encontradas": len(anomalias_prescricao)}

    # --- Vídeo (opcional: só roda se um arquivo real for informado) ---
    resultado_video = None
    if caminho_video:
        from src.video.video_pipeline import processar_video_clinico
        resultado_video = processar_video_clinico(caminho_video)
        detalhes["video"] = resultado_video

    # --- Áudio (opcional: só roda se um arquivo real for informado) ---
    resultado_audio = None
    if caminho_audio:
        from src.audio.audio_pipeline import processar_audio_consulta
        resultado_audio = processar_audio_consulta(caminho_audio)
        detalhes["audio"] = resultado_audio

    # --- Fusão multimodal + alertas ---
    resultado_risco = calcular_risco_paciente(
        paciente_id=paciente_id,
        gerenciador_alertas=gerenciador_alertas,
        anomalias_vitais=anomalias_vitais,
        anomalias_prescricao=anomalias_prescricao,
        resultado_video=resultado_video,
        resultado_audio=resultado_audio,
    )

    relatorio = gerar_relatorio(paciente_id, resultado_risco, detalhes)

    print(f"\n=== Paciente {paciente_id}: risco {relatorio['nivel_risco'].upper()} ({relatorio['pontuacao_risco']} pts) ===")
    for motivo in relatorio["motivos"]:
        print(f" - {motivo}")
    print(f"\nAlertas registrados: {gerenciador_alertas.resumo_por_severidade()}")

    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Monitoramento multimodal de pacientes (Tech Challenge Fase 4)")
    parser.add_argument("--paciente", default="paciente-001", help="Identificador do paciente")
    parser.add_argument("--video", default=None, help="Caminho de um vídeo clínico real (opcional)")
    parser.add_argument("--audio", default=None, help="Caminho de um áudio de consulta real (opcional)")
    parser.add_argument("--dados-reais", action="store_true", help="Tenta baixar sinais vitais reais do PhysioNet")
    parser.add_argument("--csv", default=None, help="Caminho de um CSV de sinais vitais (ex: amostras/sinais_vitais_exemplo.csv)")
    parser.add_argument("--demo", action="store_true", help="Roda a demonstração completa (equivalente a não passar nada)")
    argumentos = parser.parse_args()

    executar_pipeline(
        paciente_id=argumentos.paciente,
        caminho_video=argumentos.video,
        caminho_audio=argumentos.audio,
        usar_dados_reais=argumentos.dados_reais,
        caminho_csv=argumentos.csv,
    )


if __name__ == "__main__":
    main()

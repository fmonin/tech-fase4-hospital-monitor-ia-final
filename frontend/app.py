"""
Frontend de demonstração (Streamlit) do monitoramento multimodal de
pacientes — Tech Challenge Fase 4.

Como executar (Windows/PowerShell):
1) Entrar na raiz do projeto:
    cd "c:\\Users\\fmoni\\OneDrive\\Documents\\FIAP_PosGraducao\\Pós IA\\Fase 4\\hospital-monitor-ia-final"

2) Ativar ambiente virtual:
    .\\.venv\\Scripts\\Activate.ps1

3) Instalar dependências (se necessário):
    python -m pip install -r requirements.txt

4) Rodar o frontend:
    streamlit run frontend/app.py

5) Abrir no navegador:
    http://localhost:8501

Se `streamlit` não for reconhecido, use:
    python -m streamlit run frontend/app.py

Estrutura: um "atendimento" é cadastrado na barra lateral (identificação do
paciente) e cada aba de análise (Sinais Vitais, Vídeo, Áudio)
guarda seu resultado na sessão. A aba **Visão Geral** junta tudo isso pela
mesma fusão multimodal usada no restante do sistema (`calcular_risco_paciente`)
e gera o relatório final (JSON + Markdown) para download — é o mesmo motor
de decisão do `src/main.py`, só que exercitado interativamente. A aba
**Treinamento** treina os três modelos ao vivo e mostra as métricas na
tela, útil para gravar a demonstração pedida no desafio.
"""

import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Permite rodar "streamlit run frontend/app.py" a partir da raiz do projeto,
# encontrando o pacote `src` mesmo sem ele estar instalado (pip install -e .).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomalies.alert_manager import GerenciadorDeAlertas
from src.anomalies.vitals_anomaly import DetectorAnomaliasVitais
from src.config import PASTA_RELATORIOS
from src.dados import carregador_vitaldb, dados_sinteticos
from src.dados.carregador_csv import carregar_sinais_vitais_de_csv
from src.fusion.multimodal_fusion import calcular_risco_paciente
from src.reports.openai_clinical_report import gerar_parecer_openai
from src.reports.report_generator import gerar_relatorio
from src.training.utils import PASTA_MODELOS, carregar_colunas_atributos

st.set_page_config(page_title="Monitoramento Multimodal de Pacientes", page_icon="🏥", layout="wide")

CORES_SEVERIDADE = {"baixa": "#4C9F70", "media": "#E8A33D", "alta": "#D64545"}
CORES_RISCO = {"baixo": "success", "medio": "warning", "alto": "error"}
PASTA_VIDEOS_ANOTADOS = PASTA_RELATORIOS / "videos_anotados"


# --------------------------------------------------------------------------
# Estado da sessão: um "atendimento" (paciente + resultados de cada aba)
# --------------------------------------------------------------------------

def _iniciar_estado():
    padroes = {
        "paciente": {},
        "resultados": {"vitais": None, "video": None, "audio": None},
    }
    for chave, valor in padroes.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def _modelo_esta_treinado(nome: str) -> bool:
    return (PASTA_MODELOS / f"{nome}.joblib").exists()


def _nome_seguro_para_arquivo(texto: str) -> str:
    nome = re.sub(r"[^A-Za-z0-9_-]+", "-", texto.strip()).strip("-_")
    return nome or "atendimento"


def _caminho_video_anotado(nome_arquivo: str) -> Path:
    PASTA_VIDEOS_ANOTADOS.mkdir(parents=True, exist_ok=True)
    paciente_id = _nome_seguro_para_arquivo(st.session_state.paciente.get("id", "paciente-demo"))
    video_id = _nome_seguro_para_arquivo(Path(nome_arquivo).stem)
    momento = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PASTA_VIDEOS_ANOTADOS / f"{paciente_id}_{momento}_{video_id}_anotado.mp4"


def _barra_lateral():
    with st.sidebar:
        st.header("Dados do atendimento")
        with st.form("form_paciente"):
            id_paciente = st.text_input("Identificação do paciente", placeholder="Ex.: paciente-001")
            nome_paciente = st.text_input("Nome ou iniciais", placeholder="Ex.: M. S.")
            responsavel = st.text_input("Médico(a) responsável", placeholder="Ex.: Dra. Ana Silva")
            observacao = st.text_area("Observação clínica", height=90, placeholder="Motivo do atendimento...")
            salvar = st.form_submit_button("Salvar atendimento", use_container_width=True)

        if salvar:
            st.session_state.paciente = {
                "id": id_paciente.strip() or "paciente-demo",
                "nome": nome_paciente.strip(),
                "responsavel": responsavel.strip(),
                "observacao": observacao.strip(),
            }
            st.success("Atendimento salvo.")

        st.divider()
        st.caption("Status das análises")
        for rotulo, chave in [("Sinais vitais", "vitais"), ("Vídeo", "video"), ("Áudio", "audio")]:
            feito = st.session_state.resultados.get(chave) is not None
            st.write(f"{'✅' if feito else '⬜'} {rotulo}")

        st.divider()
        st.caption("Status dos modelos treinados")
        for rotulo, nome_modelo in [("Sinais vitais", "vitais_rf"), ("Movimento", "movimento_rf"), ("Voz", "audio_rf")]:
            st.write(f"{'✅' if _modelo_esta_treinado(nome_modelo) else '⚠️'} {rotulo}")

        if st.button("Limpar atendimento", use_container_width=True):
            st.session_state.paciente = {}
            st.session_state.resultados = {"vitais": None, "video": None, "audio": None}
            st.rerun()


def _cabecalho():
    st.title("🏥 Monitoramento Multimodal de Pacientes")
    st.caption("Tech Challenge Fase 4 — vídeo, áudio e sinais vitais, com IA de ponta a ponta.")
    if not st.session_state.paciente:
        st.info("Preencha e salve os dados do atendimento na barra lateral antes de gerar o relatório final "
                "(as análises individuais funcionam de qualquer forma).")
    else:
        p = st.session_state.paciente
        st.markdown(f"**Paciente:** {p.get('nome') or p.get('id')}  ·  **ID:** {p.get('id')}  ·  "
                    f"**Responsável:** {p.get('responsavel') or 'não informado'}")


# --------------------------------------------------------------------------
# Aba: Sinais vitais
# --------------------------------------------------------------------------

def _aba_sinais_vitais():
    st.subheader("Sinais vitais")
    st.write("Frequência cardíaca, SpO2 e pressão arterial, analisados minuto a minuto.")

    fonte = st.radio(
        "Fonte dos dados",
        ["Demonstração (sintético, com anomalias propositais)", "Caso real do VitalDB", "Enviar arquivo CSV"],
        horizontal=True,
    )

    arquivo_csv = None
    if fonte == "Enviar arquivo CSV":
        usar_exemplo = st.checkbox("Usar o exemplo incluído (amostras/sinais_vitais_exemplo.csv)", value=True)
        if not usar_exemplo:
            arquivo_csv = st.file_uploader(
                "Selecione o CSV (colunas: timestamp, heart_rate/frequencia_cardiaca, spo2, "
                "systolic_bp/pressao_sistolica, diastolic_bp/pressao_diastolica)",
                type=["csv"],
            )

    if st.button("Analisar sinais vitais", type="primary"):
        with st.spinner("Carregando dados e rodando o modelo treinado..."):
            if fonte.startswith("Demonstração"):
                sinais = dados_sinteticos.gerar_sinais_vitais_sinteticos(horas=12)
                st.caption("Dados sintéticos gerados agora, com duas anomalias propositais inseridas.")
            elif fonte == "Enviar arquivo CSV":
                try:
                    caminho = "amostras/sinais_vitais_exemplo.csv" if usar_exemplo else arquivo_csv
                    if caminho is None:
                        st.warning("Selecione um arquivo CSV.")
                        st.stop()
                    sinais = carregar_sinais_vitais_de_csv(caminho)
                    st.success(f"CSV carregado: {len(sinais)} amostra(s).")
                except (ValueError, FileNotFoundError) as erro:
                    st.error(f"Não foi possível ler o CSV: {erro}")
                    st.stop()
            else:
                try:
                    sinais = carregador_vitaldb.carregar_caso_real()
                    st.success("Caso real do VitalDB carregado com sucesso.")
                except ConnectionError as erro:
                    st.warning(f"Não foi possível baixar o VitalDB agora ({erro}). Usando dados sintéticos.")
                    sinais = dados_sinteticos.gerar_sinais_vitais_sinteticos(horas=12)

            anomalias = DetectorAnomaliasVitais().detectar(sinais)

        st.session_state.resultados["vitais"] = anomalias
        st.session_state["_vitais_df"] = sinais

    if st.session_state.resultados.get("vitais") is not None:
        _grafico_vitais(st.session_state["_vitais_df"], st.session_state.resultados["vitais"])
        _tabela_episodios(st.session_state.resultados["vitais"])


def _grafico_vitais(sinais: pd.DataFrame, anomalias: list[dict]):
    figura = go.Figure()
    for coluna, nome in [
        ("frequencia_cardiaca", "FC (bpm)"), ("spo2", "SpO2 (%)"),
        ("pressao_sistolica", "PA sistólica"), ("pressao_diastolica", "PA diastólica"),
    ]:
        if coluna in sinais.columns:
            figura.add_trace(go.Scatter(x=sinais["timestamp"], y=sinais[coluna], mode="lines", name=nome))

    for episodio in anomalias:
        figura.add_vrect(
            x0=episodio["inicio"], x1=episodio["fim"],
            fillcolor=CORES_SEVERIDADE.get(episodio["severidade"], "gray"), opacity=0.2, line_width=0,
        )

    figura.update_layout(height=420, margin=dict(t=20, b=20), legend=dict(orientation="h"))
    st.plotly_chart(figura, use_container_width=True)


def _tabela_episodios(anomalias: list[dict]):
    if not anomalias:
        st.success("Nenhuma anomalia encontrada no período analisado.")
        return

    st.write(f"**{len(anomalias)} episódio(s) de anomalia encontrado(s):**")
    linhas = [{
        "Métrica": e["metrica"], "Início": e["inicio"], "Fim": e["fim"],
        "Duração (min)": e["duracao_minutos"], "Ocorrências": e["ocorrencias"],
        "Origem": e["origem_deteccao"], "Severidade": e["severidade"],
    } for e in anomalias]
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Aba: Vídeo
# --------------------------------------------------------------------------

def _aba_video():
    st.subheader("Vídeo clínico (fisioterapia / cirurgia)")
    st.write("Envie um vídeo real: o sistema extrai os 33 pontos-chave do corpo (MediaPipe), calcula "
             "assimetrias/ângulos/inclinação de tronco, detecta objetos/pessoas com YOLOv8 e classifica "
             "cada janela de movimento com o modelo treinado.")

    arquivo = st.file_uploader("Arquivo de vídeo", type=["mp4", "avi", "mov", "mkv"])
    gerar_anotado = st.checkbox("Gerar vídeo anotado (esqueleto + caixas + alertas)", value=True)

    if arquivo and st.button("Processar vídeo", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(arquivo.name).suffix) as tmp:
            tmp.write(arquivo.read())
            caminho_temporario = tmp.name

        caminho_saida = str(_caminho_video_anotado(arquivo.name)) if gerar_anotado else None

        try:
            with st.spinner("Extraindo postura, detectando objetos e classificando movimento... pode levar alguns minutos."):
                from src.video.video_pipeline import processar_video_clinico
                resultado = processar_video_clinico(caminho_temporario, caminho_video_anotado=caminho_saida)
        except Exception as erro:
            st.error(f"Não foi possível processar o vídeo: {erro}")
            st.caption("Confirme se `mediapipe` e `ultralytics` estão instalados (ver requirements.txt).")
            return

        st.session_state.resultados["video"] = resultado

    resultado = st.session_state.resultados.get("video")
    if resultado:
        colunas = st.columns(3)
        colunas[0].metric("Frames analisados", resultado["total_frames_analisados"])
        colunas[1].metric("Eventos de movimento anômalo", len(resultado["anomalias_movimento"]))
        colunas[2].metric("Objetos/pessoas detectados", sum(resultado["objetos_detectados"].values()))

        if resultado["possivel_desvio_no_procedimento"]:
            st.warning("⚠️ Possível desvio no procedimento: eventos de movimento fora do padrão foram detectados.")
        else:
            st.success("Nenhum desvio de movimento detectado — padrão consistente com uma sessão normal.")

        if resultado.get("video_anotado"):
            st.write("**Vídeo anotado (esqueleto, objetos detectados e alertas):**")
            st.video(resultado["video_anotado"])
            caminho_anotado = Path(resultado["video_anotado"])
            st.caption(f"Salvo em: `{caminho_anotado.relative_to(PASTA_RELATORIOS.parent)}`")
            if caminho_anotado.exists():
                st.download_button(
                    "Baixar vídeo anotado",
                    data=caminho_anotado.read_bytes(),
                    file_name=caminho_anotado.name,
                    mime="video/mp4",
                )
        elif resultado.get("aviso_video_anotado"):
            st.warning(f"O vídeo anotado não foi gerado: {resultado['aviso_video_anotado']}")

        if resultado["objetos_detectados"]:
            st.write("**Objetos/pessoas detectados no vídeo:**")
            st.dataframe(pd.DataFrame(resultado["objetos_detectados"].items(), columns=["Classe", "Ocorrências"]),
                         use_container_width=True, hide_index=True)

        if resultado["anomalias_movimento"]:
            st.write("**Eventos de movimento anômalo (por janela de frames), com o motivo clínico:**")
            eventos = pd.DataFrame(resultado["anomalias_movimento"])
            if "motivos" in eventos.columns:
                eventos["motivos"] = eventos["motivos"].apply(lambda m: ", ".join(m) if m else "—")
            st.dataframe(eventos, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Aba: Vídeos gerados
# --------------------------------------------------------------------------

def _aba_videos():
    st.subheader("Vídeos anotados")
    st.write("Reveja ou baixe os vídeos gerados após a análise de movimento.")

    if not PASTA_VIDEOS_ANOTADOS.exists():
        st.info("Nenhum vídeo anotado foi gerado ainda.")
        return

    videos = sorted(
        PASTA_VIDEOS_ANOTADOS.glob("*.mp4"),
        key=lambda caminho: caminho.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        st.info("Nenhum vídeo anotado foi gerado ainda.")
        return

    video_escolhido = st.selectbox(
        "Vídeo disponível",
        videos,
        format_func=lambda caminho: caminho.name,
    )
    st.caption(f"{len(videos)} vídeo(s) persistido(s) em `reports/videos_anotados/`.")
    st.video(str(video_escolhido))
    st.download_button(
        "Baixar vídeo selecionado",
        data=video_escolhido.read_bytes(),
        file_name=video_escolhido.name,
        mime="video/mp4",
    )


# --------------------------------------------------------------------------
# Aba: Áudio
# --------------------------------------------------------------------------

def _aba_audio():
    st.subheader("Áudio de consulta médica")
    st.write("Envie um áudio real: o sistema extrai características acústicas, classifica indícios de "
             "fadiga/disartria com o modelo treinado, identifica eventos sonoros com YAMNet (treinado no "
             "AudioSet) e, se o Azure estiver configurado, transcreve e analisa o texto.")

    arquivo = st.file_uploader("Arquivo de áudio", type=["wav", "mp3", "m4a", "ogg"])
    if arquivo:
        st.audio(arquivo.getvalue())

    if "_audio_status_azure" not in st.session_state:
        st.session_state["_audio_status_azure"] = []

    if arquivo and st.button("Processar áudio", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(arquivo.name).suffix) as tmp:
            tmp.write(arquivo.getbuffer())
            caminho_temporario = tmp.name

        st.session_state["_audio_status_azure"] = []

        status_box = st.empty()

        def _atualizar_status_azure(mensagem: str):
            st.session_state["_audio_status_azure"].append(mensagem)
            status_box.info("\n".join(f"- {item}" for item in st.session_state["_audio_status_azure"]))

        try:
            with st.spinner("Extraindo características acústicas e classificando..."):
                from src.audio.audio_pipeline import processar_audio_consulta
                resultado = processar_audio_consulta(caminho_temporario, status_callback=_atualizar_status_azure)
        except Exception as erro:
            st.error(f"Não foi possível processar o áudio: {erro}")
            return

        st.session_state.resultados["audio"] = resultado

    resultado = st.session_state.resultados.get("audio")
    if resultado:
        status_azure = st.session_state.get("_audio_status_azure") or []
        if status_azure:
            st.write("**Etapas Azure executadas:**")
            for item in status_azure:
                st.write(f"• {item}")

        indicadores = resultado["indicadores_acusticos"]
        colunas = st.columns(4)
        colunas[0].metric("Duração", f"{indicadores['duracao_segundos']}s")
        colunas[1].metric("Proporção de silêncio", f"{indicadores['proporcao_silencio']*100:.0f}%")
        colunas[2].metric("Variabilidade de pitch", f"{indicadores['variabilidade_pitch_hz']} Hz")
        colunas[3].metric("Energia média", indicadores["energia_media"])

        classificacao = resultado.get("classificacao_modelo_treinado")
        if classificacao:
            cor = "🔴" if classificacao["classe"] == "alterado" else "🟢"
            st.write(f"{cor} **Classificação do modelo treinado:** {classificacao['classe']} "
                     f"(confiança {classificacao['confianca']*100:.0f}%)")
        else:
            st.info("Modelo treinado de áudio não encontrado — usando apenas os indicadores heurísticos acima.")

        eventos_sonoros = resultado.get("eventos_sonoros_audioset")
        if eventos_sonoros:
            st.write("**Eventos sonoros identificados (YAMNet, treinado no AudioSet):**")
            df_eventos = pd.DataFrame(eventos_sonoros).rename(
                columns={"classe": "Classe", "pontuacao": "Confiança", "relevante_clinicamente": "Relevância clínica"})
            st.dataframe(df_eventos, use_container_width=True, hide_index=True)

        if resultado.get("transcricao"):
            st.write("**Transcrição (Azure Speech to Text):**")
            st.write(resultado["transcricao"])
            analise = resultado.get("analise_texto") or {}
            if analise:
                termos_criticos = analise.get("termos_criticos_encontrados") or []
                frases_chave = analise.get("frases_chave") or []
                confianca = analise.get("confianca_sentimento") or {}
                st.write("**Análise feita pela API Azure Text Analytics (descrição escrita):**")
                st.write(
                    "A API AZure 'Text Analytics' avaliou a transcrição e classificou o sentimento geral como "
                    f"**{analise.get('sentimento', 'indefinido')}**. "
                    f"Pontuações de confiança: positivo {confianca.get('positivo', 0):.2f}, "
                    f"neutro {confianca.get('neutro', 0):.2f} e negativo {confianca.get('negativo', 0):.2f}. "
                    f"Termos críticos identificados: {', '.join(termos_criticos) if termos_criticos else 'nenhum'}. "
                    f"Principais frases-chave extraídas: {', '.join(frases_chave[:5]) if frases_chave else 'não identificadas'}."
                )

                trechos_sentimento = analise.get("trechos_sentimento") or []
                if trechos_sentimento:
                    trechos_negativos = [
                        item for item in trechos_sentimento
                        if item.get("sentimento", "").lower() == "negative"
                    ]

                    if trechos_negativos:
                        st.write("**Trechos com sentimento negativo (destaque clínico):**")
                        for item in trechos_negativos:
                            confianca_trecho = item.get("confianca") or {}
                            texto_trecho = (item.get("trecho") or "").strip()
                            destaque = (
                                "Sentimento: negative | "
                                "Confiança (pos/neut/neg): "
                                f"{confianca_trecho.get('positivo', 0):.2f}/"
                                f"{confianca_trecho.get('neutro', 0):.2f}/"
                                f"{confianca_trecho.get('negativo', 0):.2f}"
                            )
                            st.error(f"{destaque}\n\n\"{texto_trecho}\"")
                    else:
                        st.success("Nenhum trecho com sentimento negativo foi identificado na transcrição.")

                    with st.expander("Ver todos os trechos com sentimento identificado"):
                        for item in trechos_sentimento:
                            sentimento_trecho = item.get("sentimento", "indefinido")
                            confianca_trecho = item.get("confianca") or {}
                            texto_trecho = (item.get("trecho") or "").strip()
                            destaque = (
                                f"Sentimento: {sentimento_trecho} | "
                                f"Confiança (pos/neut/neg): "
                                f"{confianca_trecho.get('positivo', 0):.2f}/"
                                f"{confianca_trecho.get('neutro', 0):.2f}/"
                                f"{confianca_trecho.get('negativo', 0):.2f}"
                            )

                            if sentimento_trecho == "negative":
                                st.error(f"{destaque}\n\n\"{texto_trecho}\"")
                            elif sentimento_trecho == "positive":
                                st.success(f"{destaque}\n\n\"{texto_trecho}\"")
                            else:
                                st.info(f"{destaque}\n\n\"{texto_trecho}\"")
        elif resultado.get("avisos"):
            st.caption("ℹ️ " + " / ".join(resultado["avisos"]))


# --------------------------------------------------------------------------
# Aba: Visão geral / relatório consolidado
# --------------------------------------------------------------------------

def _aba_visao_geral():
    st.subheader("Visão geral e relatório consolidado")
    st.write("Junta os resultados de todas as abas já analisadas pela mesma fusão multimodal usada em "
             "`src/main.py`, e gera o relatório final do atendimento.")

    resultados = st.session_state.resultados
    concluidas = [chave for chave, valor in resultados.items() if valor is not None]
    st.caption(f"Modalidades analisadas nesta sessão: {', '.join(concluidas) if concluidas else 'nenhuma ainda'}")

    if not concluidas:
        st.info("Analise ao menos uma modalidade (Sinais Vitais, Vídeo ou Áudio) para gerar o relatório.")
        return

    if st.button("Gerar relatório consolidado", type="primary"):
        paciente_id = st.session_state.paciente.get("id") or "paciente-demo"
        gerenciador = GerenciadorDeAlertas()

        resultado_risco = calcular_risco_paciente(
            paciente_id=paciente_id,
            gerenciador_alertas=gerenciador,
            anomalias_vitais=resultados.get("vitais"),
            resultado_video=resultados.get("video"),
            resultado_audio=resultados.get("audio"),
        )

        detalhes = {
            "paciente": st.session_state.paciente,
            "vitais": resultados.get("vitais"),
            "video": resultados.get("video"),
            "audio": resultados.get("audio"),
        }

        try:
            resultado_risco["resumo_openai"] = gerar_parecer_openai(
                paciente_id=paciente_id,
                resultado_risco=resultado_risco,
                detalhes=detalhes,
            )
        except RuntimeError as erro:
            st.warning(str(erro))
            resultado_risco["resumo_openai"] = None
        except Exception as erro:
            st.warning(f"Falha inesperada ao gerar parecer OpenAI: {erro}")
            resultado_risco["resumo_openai"] = None

        relatorio = gerar_relatorio(paciente_id, resultado_risco, detalhes)
        st.session_state["_ultimo_relatorio"] = relatorio
        st.session_state["_ultimo_resumo_alertas"] = gerenciador.resumo_por_severidade()

    relatorio = st.session_state.get("_ultimo_relatorio")
    if relatorio:
        nivel = relatorio["nivel_risco"]
        getattr(st, CORES_RISCO.get(nivel, "info"))(f"Classificação: {nivel.upper()} ({relatorio['pontuacao_risco']} pontos)")

        resumo_openai = relatorio.get("resumo_openai")
        if resumo_openai:
            st.subheader("Parecer Final (OpenAI)")
            colunas = st.columns(2)
            colunas[0].metric("Pontuação do paciente (0-10)", resumo_openai["pontuacao_paciente_0a10"])
            colunas[1].metric("Item de maior risco", resumo_openai["item_maior_risco"])

            st.error(f"Alerta para equipe médica: {resumo_openai['alerta_equipe_medica']}")
            if resumo_openai.get("justificativa_curta"):
                st.caption(f"Justificativa: {resumo_openai['justificativa_curta']}")

        st.write("**Motivos identificados:**")
        if relatorio["motivos"]:
            for motivo in relatorio["motivos"]:
                st.write(f"• {motivo}")
        else:
            st.write("Nenhuma anomalia relevante identificada.")

        resumo = st.session_state.get("_ultimo_resumo_alertas", {})
        colunas = st.columns(3)
        colunas[0].metric("Alertas de baixa severidade", resumo.get("baixa", 0))
        colunas[1].metric("Alertas de média severidade", resumo.get("media", 0))
        colunas[2].metric("Alertas de alta severidade", resumo.get("alta", 0))

        import json
        conteudo_json = json.dumps(relatorio, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "Baixar relatório JSON", data=conteudo_json,
            file_name=f"relatorio_{relatorio['paciente_id']}.json", mime="application/json",
        )
        with st.expander("Ver relatório completo"):
            st.json(relatorio)


# --------------------------------------------------------------------------
# Aba: Treinamento
# --------------------------------------------------------------------------

def _aba_treinamento():
    st.subheader("Treinamento dos modelos")
    st.write(
        "Cada modelo é um `RandomForestClassifier` treinado sobre um conjunto de dados rotulado (ver "
        "`src/training/`). Clique para treinar e ver as métricas — conjunto de dados construído, divisão "
        "treino/teste, e avaliação, tudo ao vivo."
    )

    nomes_modelos = {
        "Sinais vitais": ("src.training.treinar_modelo_vitais", "treinar", "vitais_rf"),
        "Movimento (vídeo)": ("src.training.treinar_modelo_movimento", "treinar", "movimento_rf"),
        "Voz (áudio)": ("src.training.treinar_modelo_audio", "treinar", "audio_rf"),
    }

    for rotulo, (modulo, funcao, nome_arquivo) in nomes_modelos.items():
        with st.expander(f"Modelo: {rotulo}", expanded=False):
            if st.button(f"Treinar modelo de {rotulo.lower()}", key=f"treinar_{nome_arquivo}"):
                with st.spinner("Construindo dataset, treinando e avaliando..."):
                    import importlib
                    mod = importlib.import_module(modulo)
                    _, metricas = getattr(mod, funcao)()

                st.success(f"Modelo treinado e salvo em models/{nome_arquivo}.joblib")
                colunas = st.columns(4)
                colunas[0].metric("Acurácia", f"{metricas['acuracia']*100:.1f}%")
                colunas[1].metric("Precisão", f"{metricas['precisao']*100:.1f}%")
                colunas[2].metric("Recall", f"{metricas['recall']*100:.1f}%")
                colunas[3].metric("F1-score", f"{metricas['f1_score']*100:.1f}%")

                matriz = metricas["matriz_confusao"]
                figura = px.imshow(
                    matriz, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Previsto", y="Real", color="Contagem"),
                    x=["normal", "anômalo"], y=["normal", "anômalo"],
                )
                figura.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(figura, use_container_width=True)

            colunas_atributos = carregar_colunas_atributos(nome_arquivo)
            if colunas_atributos:
                st.caption(f"Atributos usados por este modelo: {', '.join(colunas_atributos)}")

    with st.expander("Modelo alternativo: sinais vitais com sinal REAL do PhysioNet (MIT-BIH)"):
        st.write(
            "Treina um `IsolationForest` não supervisionado usando frequência cardíaca REAL, derivada das "
            "anotações de batimento do MIT-BIH Arrhythmia Database. Complementar ao modelo supervisionado "
            "acima (que usa dados sintéticos rotulados). Requer acesso à internet a physionet.org."
        )
        if st.button("Treinar com sinal real do PhysioNet", key="treinar_physionet"):
            try:
                with st.spinner("Baixando registro real do PhysioNet e treinando..."):
                    from src.training.treinar_modelo_vitais_physionet import treinar as treinar_physionet
                    _, info = treinar_physionet()
                st.success(f"Treinado com {info['n_amostras']} amostras reais; "
                           f"{info['n_anomalias']} anomalias identificadas na própria série.")
            except ConnectionError as erro:
                st.warning(f"Sem acesso ao PhysioNet agora: {erro}")

    with st.expander("Modelo alternativo: movimento com vídeos REAIS de postura normal"):
        st.write(
            "Treina um `IsolationForest` não supervisionado sobre janelas de postura extraídas de vídeos "
            "reais colocados em `amostras/videos_normais/` — o modelo aprende como é o padrão normal e "
            "sinaliza qualquer coisa fora disso, sem precisar de um exemplo real de queda/espasmo rotulado. "
            "Já vêm vídeos de exemplo prontos nessa pasta; também é possível baixar vídeos reais de "
            "fisioterapia do Kaggle (dataset `toobasaeed11/physiotherapy`, requer conta e autenticação)."
        )
        col_kaggle, col_treinar = st.columns(2)
        if col_kaggle.button("Baixar vídeos reais do Kaggle", key="baixar_kaggle"):
            try:
                with st.spinner("Baixando dataset do Kaggle (pode levar alguns minutos)..."):
                    from src.dados.baixar_kaggle_fisioterapia import baixar as baixar_kaggle
                    manifesto = baixar_kaggle()
                st.success(f"{manifesto['total_videos']} vídeo(s) baixado(s) e prontos para treino.")
            except SystemExit as erro:
                st.warning(str(erro))

        if col_treinar.button("Treinar com vídeos reais", key="treinar_movimento_real"):
            from src.config import PASTA_AMOSTRAS
            pasta_videos = PASTA_AMOSTRAS / "videos_normais"
            try:
                with st.spinner("Extraindo postura dos vídeos e treinando..."):
                    from src.training.treinar_modelo_movimento_real import treinar as treinar_movimento_real
                    _, info = treinar_movimento_real()
                st.success(f"Treinado com {info['n_videos']} vídeo(s) reais, {info['n_janelas']} janelas; "
                           f"{info['n_fora_do_padrao']} janelas fora do padrão na própria série.")
            except SystemExit as erro:
                st.warning(f"{erro}\n\nColoque vídeos em `{pasta_videos}` e tente novamente.")

    with st.expander("Modelo alternativo: voz com áudios REAIS (embeddings do YAMNet)"):
        st.write(
            "Treina um `RandomForestClassifier` sobre embeddings do YAMNet (transfer learning a partir do "
            "AudioSet) extraídos de áudios reais organizados por classe em `amostras/audio_por_classe/"
            "<classe>/`. Requer pelo menos 2 classes com alguns áudios cada (ex: `normal/`, `alterado/`)."
        )
        if st.button("Treinar com áudios reais", key="treinar_audio_real"):
            from src.config import PASTA_AMOSTRAS
            pasta_audios = PASTA_AMOSTRAS / "audio_por_classe"
            try:
                with st.spinner("Extraindo embeddings do YAMNet e treinando..."):
                    from src.training.treinar_modelo_audio_real import treinar as treinar_audio_real
                    _, info = treinar_audio_real()
                st.success(f"Treinado com {info['n_audios']} áudio(s) reais; classes: {', '.join(info['classes'])}.")
            except SystemExit as erro:
                st.warning(f"{erro}\n\nOrganize áudios em `{pasta_audios}/<classe>/` e tente novamente.")


def main():
    _iniciar_estado()
    _barra_lateral()
    _cabecalho()

    aba_geral, aba_vitais, aba_video, aba_videos, aba_audio, aba_treino = st.tabs(
        ["📋 Visão Geral", "📈 Sinais Vitais", "🎥 Vídeo", "🎞️ Vídeos", "🎙️ Áudio", "🧠 Treinamento"]
    )
    with aba_geral:
        _aba_visao_geral()
    with aba_vitais:
        _aba_sinais_vitais()
    with aba_video:
        _aba_video()
    with aba_videos:
        _aba_videos()
    with aba_audio:
        _aba_audio()
    with aba_treino:
        _aba_treinamento()

    st.divider()
    st.caption("Protótipo acadêmico — Tech Challenge Fase 4. Não substitui diagnóstico ou decisão clínica.")


if __name__ == "__main__":
    main()

# Relatório Técnico — Monitoramento Multimodal de Pacientes com IA

**Projeto:** Tech Challenge FIAP — Fase 4  
**Objetivo:** integrar vídeo, áudio e sinais vitais para identificar situações fora do padrão, consolidar o risco do paciente e apresentar achados explicáveis à equipe.

> Este é um protótipo acadêmico. Os resultados apoiam uma demonstração técnica e não substituem avaliação, diagnóstico ou decisão clínica profissional.

## 1. Visão geral da solução

O projeto trabalha com três modalidades de dados:

- **Sinais vitais:** frequência cardíaca, SpO2 e pressão arterial.
- **Vídeo:** postura corporal e objetos/pessoas presentes na cena.
- **Áudio:** características da voz, eventos sonoros e, quando o Azure está configurado, transcrição e análise do texto.

A saída de cada modalidade é transformada em eventos de risco. Em vez de misturar dados brutos tão diferentes em um único classificador, o sistema usa **fusão tardia**: cada modalidade é interpretada pelo seu módulo especializado e só depois os resultados entram no cálculo do risco consolidado.

## 2. Fluxo multimodal

```text
Sinais vitais                  Vídeo                         Áudio
FC, SpO2 e pressão             Arquivo clínico               Áudio de consulta
        |                           |                              |
Regras clínicas +              MediaPipe Pose +              librosa + YAMNet +
RandomForest                   YOLOv8 + RandomForest         Azure (opcional)
        |                           |                              |
Episódios de anomalia          Eventos de movimento           Alteração de voz,
com início, fim e duração      e vídeo anotado                termos e sons relevantes
        \                           |                              /
         \____________________ FUSÃO TARDIA ____________________/
                                  |
                         Pontuação de risco
                                  |
                   Alertas e relatório JSON/Markdown
```

### Etapas da fusão

1. Cada pipeline recebe e valida seu próprio tipo de arquivo ou série temporal.
2. O pipeline extrai atributos e detecta anomalias ou sinais de atenção.
3. A fusão recebe somente os achados finais: episódios de vitais, eventos de movimento, classificação de voz, termos críticos e eventos sonoros.
4. Cada achado recebe um peso conforme sua origem e gravidade.
5. A pontuação resultante é classificada em risco **baixo**, **médio** ou **alto** e é registrada junto dos motivos que explicam o resultado.

Essa separação permite analisar um paciente mesmo quando alguma modalidade está ausente, como em um atendimento que tenha apenas CSV de sinais vitais.

## 3. Modelos e técnicas por tipo de dado

| Modalidade | Técnica ou modelo | Aplicação no projeto |
|---|---|---|
| Sinais vitais | Regras de limites clínicos | Detecta valores fisiológicos fora dos limites configurados, como SpO2 baixa, taquicardia ou pressão elevada. |
| Sinais vitais | `vitais_rf` — RandomForestClassifier | Analisa janelas de 15 minutos com média, desvio, mínimo e máximo dos sinais para identificar combinações anormais. |
| Sinais vitais | IsolationForest opcional | Modelo alternativo não supervisionado treinável a partir de série real do PhysioNet/MIT-BIH. |
| Vídeo | MediaPipe Pose / PoseLandmarker | Extrai 33 pontos corporais por frame para formar o esqueleto e calcular atributos posturais. |
| Vídeo | YOLOv8n | Detecta pessoas e objetos na cena; as caixas encontradas são desenhadas no vídeo anotado. |
| Vídeo | `movimento_rf` — RandomForestClassifier | Classifica janelas de 15 frames usando assimetria de ombros/quadril, inclinação do tronco, ângulos de joelhos/cotovelos e velocidade. |
| Vídeo | IsolationForest opcional | Aprende o padrão de vídeos reais de movimento normal para indicar janelas fora do padrão. |
| Áudio | `audio_rf` — RandomForestClassifier | Classifica indícios de alteração na voz a partir de variabilidade de pitch, proporção de silêncio e energia média. |
| Áudio | Azure Speech to Text | Transcreve a fala da consulta quando as credenciais estão configuradas. |
| Áudio | Azure Text Analytics | Avalia sentimento por trecho, frases-chave e termos críticos da transcrição. |
| Áudio | YAMNet pré-treinado no AudioSet | Indica eventos sonoros, como tosse, gemido, engasgo ou respiração, quando as dependências opcionais estão disponíveis. |

### Vídeo anotado e explicabilidade

Quando a opção de gerar vídeo anotado é selecionada, o sistema salva um MP4 em `reports/videos_anotados/`. O arquivo contém:

- o esqueleto obtido pelo MediaPipe;
- caixas e rótulos do YOLOv8;
- painel vermelho `ALERTA: MOVIMENTO FORA DO PADRÃO` nos frames anômalos;
- marcações vermelhas nas articulações relacionadas ao motivo identificado, como ombros, quadril, tronco, joelhos ou cotovelos.

O MP4 é convertido para H.264 antes de ser apresentado nas abas **Vídeo** e **Vídeos** do frontend, para ser compatível com o player do navegador.

## 4. Treinamento e métricas disponíveis

Os modelos supervisionados são treinados com `RandomForestClassifier` e separados do pipeline de atendimento. O treinamento constrói o conjunto de dados, separa treino e teste, mede acurácia, precisão, recall e F1-score, e salva o modelo em `models/`.

As métricas abaixo correspondem aos artefatos presentes no repositório:

| Modelo | Acurácia | Precisão | Recall | F1-score | Dados e atributos |
|---|---:|---:|---:|---:|---|
| `movimento_rf` | 96,0% | 95,1% | 97,0% | 96,0% | Janelas de postura com 16 atributos agregados de assimetria, ângulos, tronco e velocidade. |
| `audio_rf` | 91,0% | 89,3% | 92,9% | 91,1% | Variabilidade de pitch, proporção de silêncio e energia média. |
| `vitais_rf` | Métricas geradas no treino | Métricas geradas no treino | Métricas geradas no treino | Métricas geradas no treino | Janelas de 15 minutos com estatísticas de FC, SpO2 e pressão. |

Os conjuntos supervisionados principais usam dados sintéticos rotulados para permitir testes controlados e repetíveis. Isso demonstra o funcionamento técnico do pipeline, mas não equivale a validar desempenho clínico em uma população real. Os caminhos alternativos com IsolationForest e dados reais existem para aproximar o projeto desse cenário quando há dados disponíveis.

## 5. Resultados e exemplos de anomalias detectadas

### 5.1 Sinais vitais

Em uma execução registrada para `paciente-demo`, foram identificados eventos como:

- **SpO2 fora do limite clínico** entre `04:00` e `04:09`, com valor de pico de 89,1%.
- **Frequência cardíaca elevada** em episódios iniciados por volta de `08:00`.
- **Pressão sistólica elevada** no mesmo intervalo.
- **Janelas classificadas como anômalas pelo modelo treinado de vitais**, incluindo episódios sustentados de 23 e 29 ocorrências.

Esse resultado mostra as duas camadas funcionando juntas: os limites clínicos detectam um valor diretamente perigoso e o modelo identifica o padrão combinado da janela temporal.

### 5.2 Movimento em vídeo

Na análise de um vídeo de abdução de ombro, o projeto registrou eventos de movimento fora do padrão. Os motivos mais frequentes no relatório foram:

- inclinação excessiva do tronco;
- velocidade de movimento alterada;
- assimetria de ombros.

O vídeo anotado apresenta essas regiões corporais destacadas durante os frames marcados pelo classificador, permitindo revisar visualmente o motivo do alerta em vez de receber apenas uma classificação abstrata.

### 5.3 Áudio

O pipeline de áudio pode produzir, conforme o arquivo enviado e as integrações configuradas:

- classificação `normal` ou `alterado` pelo modelo de voz;
- proporção de silêncio, energia e variabilidade de pitch;
- sentimento geral e sentimento por trecho da transcrição;
- termos críticos como relatos de dor ou falta de ar;
- eventos sonoros relevantes sugeridos pelo YAMNet.

Um termo crítico identificado pelo Text Analytics ou um evento sonoro grave indicado pelo YAMNet aumenta a pontuação de risco da fusão e é incluído no relatório consolidado.

## 6. Resultado consolidado e alertas

A função `calcular_risco_paciente` consolida os achados. Os pesos privilegiam situações que exigem atenção imediata, como limites clínicos violados, termos críticos na fala e eventos sonoros graves. A classificação atual usa:

| Pontuação | Classificação |
|---:|---|
| 0 a 2 | Baixo |
| 3 a 7 | Médio |
| 8 ou mais | Alto |

Além do nível final, o relatório lista os motivos que levaram à pontuação. Essa lista é importante para tornar o alerta auditável e facilitar a revisão humana.

## 7. Limitações e próximos passos

- As métricas de modelos treinados com dados sintéticos não representam validação clínica real.
- O YOLOv8n identifica classes gerais do COCO; para instrumentos médicos, seria necessário novo treinamento com imagens rotuladas do domínio.
- O projeto deve ser validado com dados reais anonimizados, protocolos clínicos e revisão de especialistas antes de qualquer uso assistencial.
- A pontuação de fusão é baseada em pesos explícitos e pode ser calibrada com dados reais e retorno da equipe clínica.

## 8. Como reproduzir

```bash
python -m src.training.treinar_tudo
python -m streamlit run frontend/app.py
pytest tests/ -v
```

Pelo frontend, é possível analisar sinais vitais, enviar vídeo e áudio, gerar o vídeo anotado, consultar os vídeos salvos e baixar o relatório final.

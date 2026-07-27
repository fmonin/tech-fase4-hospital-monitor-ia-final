# Relatório Técnico — Monitoramento Multimodal de Pacientes

Tech Challenge Fase 4 · Pós-Tech

## 1. Contexto e objetivo

O hospital do desafio já usa IA para analisar exames e laudos e quer avançar
para monitoramento **contínuo** de pacientes, cruzando três tipos de dado —
vídeo, áudio e séries temporais clínicas — para identificar sinais precoces
de risco. Este projeto implementa esse monitoramento fim a fim: da ingestão
de cada modalidade até o alerta final para a equipe médica.

## 2. Fluxo multimodal

O desenho é de **fusão tardia** (late fusion): cada modalidade é processada
e analisada por um módulo especializado, e só depois os resultados são
combinados num pontuação de risco único.

```
┌───────────────┐   ┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐
│     Vídeo     │   │     Áudio     │   │   Sinais vitais     │   │   Prescrições     │
│ (fisioterapia/│   │  (consultas)  │   │ (FC, SpO2, pressão)  │   │ (evolução do       │
│   cirurgia)   │   │               │   │                      │   │  tratamento)       │
└───────┬───────┘   └───────┬───────┘   └──────────┬───────────┘   └─────────┬─────────┘
        │                   │                       │                        │
   MediaPipe Pose      Azure Speech            Limites clínicos        Regras de variação
   + YOLOv8            + Text Analytics        + modelo treinado       de dose/frequência
   (postura/objetos)   + acústica (librosa)     (RandomForest)          (evolução discreta)
        │                   │                       │                        │
   Modelo treinado     Modelo treinado         Episódios de              Alterações
   (movimento,         (voz: fadiga/           anomalia (com             fora do padrão
   RandomForest)        disartria)              início/fim/duração)
        │                   │                       │                        │
        └───────────────────┴───────────┬───────────┴────────────────────────┘
                                         ▼
                          FUSÃO MULTIMODAL (pontuação de risco)
                                         ▼
                    GERENCIADOR DE ALERTAS → EQUIPE MÉDICA
                                         ▼
                      RELATÓRIO AUTOMÁTICO (JSON + Markdown)
```

Os três blocos "modelo treinado" (sinais vitais, movimento e voz) não são
ajustados durante a execução do pipeline: eles são treinados uma única vez
por scripts próprios em `src/training/`, avaliados num conjunto de teste
separado, e salvos em `models/*.joblib`. O pipeline de inferência (o que
roda a cada paciente/vídeo/áudio) só **carrega** esses arquivos — a seção 4
detalha como cada um foi treinado.

**Por que fusão tardia, e não um único modelo com todas as atributos
misturadas?** Vídeo, áudio e séries temporais têm naturezas estatísticas
completamente diferentes (imagens vs. ondas sonoras vs. números por minuto).
Um modelo único exigiria uma arquitetura de deep learning complexa e um
volume de dados rotulados que não existe para esse domínio específico. A
fusão tardia permite usar, em cada modalidade, a técnica mais adequada e
mais simples de validar — e permite também rodar o sistema parcialmente
(ex: só com vitais, sem vídeo daquele dia) sem que ele pare de funcionar.

## 3. Modelos e técnicas aplicados por tipo de dado

### 3.1 Vídeo

| Etapa | Técnica | Observação |
|---|---|---|
| Extração de postura | **MediaPipe Pose** | Substitui o OpenPose sugerido no enunciado: mesma finalidade (33 pontos-chave do corpo por frame, corpo inteiro — não só um ponto isolado), instalação muito mais simples (sem compilação C++/CUDA), o que a torna viável em qualquer ambiente. |
| Features clínicas explicáveis | `src/video/pose_features.py` | A partir dos 33 pontos, calcula 8 atributos com significado clínico direto: assimetria de ombros e de quadril, inclinação do tronco, ângulos de joelho e cotovelo (esquerdo/direito), e velocidade de movimento — mais próximo do que um fisioterapeuta observaria do que "a velocidade de um ponto". Cada evento de anomalia já sai com os "motivos" (ex: "assimetria de ombros", "inclinação excessiva do tronco"). |
| Detecção de objetos/pessoas em cena | **YOLOv8** (modelo `yolov8n`, pré-treinado em COCO) | Usado como sugerido no enunciado. Detecta pessoas em quadro (ex: confirmar presença contínua do paciente/fisioterapeuta). Arquitetura pronta para receber um YOLOv8 re-treinado com um conjunto de dados de instrumentos/objetos médicos, quando disponível. |
| Detecção de movimento anômalo | **Modelo treinado** (RandomForest sobre média/desvio das 8 atributos de pose, por janela) | Ver seção 4.2. Fallback: regra por limiar sobre as mesmas 8 atributos (`pose_features.motivos_da_regra`), usada só se o modelo treinado ainda não existir em disco. |
| Vídeo anotado | `src/video/video_anotado.py` | Gera uma cópia do vídeo com o esqueleto do MediaPipe desenhado, as caixas do YOLOv8 e um aviso em vermelho nos frames que o modelo de movimento classificou como anômalos — camada de apresentação, exibida na aba "Vídeo" do frontend. |

### 3.2 Áudio

| Etapa | Técnica | Observação |
|---|---|---|
| Transcrição | **Azure Speech to Text** | Conforme pedido no enunciado. Requer credenciais Azure (`AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION`). |
| Sentimento e termos críticos | **Azure Text Analytics** | Sentimento geral da fala + frases-chave, cruzados com uma lista de termos clínicos de atenção (ex: "dor no peito", "falta de ar"). |
| Indícios de fadiga/disartria | Características acústicas via **librosa** + **modelo treinado** (RandomForest) | Ver seção 4.3. As características (pitch, pausas, energia) alimentam o classificador treinado; se ele não existir em disco, cai para uma pontuação heurística simples sobre as mesmas características. |
| Eventos sonoros | **YAMNet** (rede treinada pelo Google diretamente no AudioSet completo, via TensorFlow Hub) | Ver seção 4.4. Identifica sons clinicamente relevantes no áudio (tosse, respiração ofegante, engasgo, gemido, choro etc.), complementando a transcrição de fala. Etapa opcional: se `tensorflow`/`tensorflow-hub` não estiverem disponíveis, o pipeline segue sem ela. |

### 3.3 Séries temporais clínicas

| Etapa | Técnica | Observação |
|---|---|---|
| Sinais vitais | Limites clínicos fixos **+** **modelo treinado** (RandomForest) | Ver seção 4.1. Duas camadas complementares: regras explicáveis para violações de limite fisiológico (ex: SpO2 < 92%), e um classificador treinado para combinações incomuns entre sinais que uma regra isolada não captura. Fallback: Isolation Forest ajustado na hora, se o modelo treinado não existir. |
| Agrupamento em episódios | Agregação de eventos consecutivos | Minutos anômalos consecutivos da mesma métrica viram um único "episódio" (início, fim, duração, nº de ocorrências) — evita alertar a equipe médica um alerta por minuto, e reflete como monitores clínicos reais reportam eventos sustentados. |
| Prescrições | Regras de variação percentual (dose e frequência) | Evento raro e discreto: preferimos uma regra 100% explicável ("a dose subiu 300% de um dia para o outro") a um modelo estatístico opaco. |

### 3.4 Fusão e alertas

Cada anomalia entra com um peso no pontuação de risco (ex: violação de limite
clínico pesa mais que um desvio estatístico isolado; um termo crítico na
fala do paciente pesa mais que um indício de fadiga). O pontuação final
classifica o paciente em risco **baixo / médio / alto**, e cada anomalia
gera um alerta individual, com severidade, registrado pelo
`GerenciadorDeAlertas` — hoje via log estruturado, com um ponto único de
extensão para plugar e-mail/SMS/painel do hospital sem tocar na lógica de
detecção.

## 4. Treinamento dos modelos — passo a passo

Os três modelos usados na detecção de anomalias (sinais vitais, movimento e
voz) seguem exatamente o mesmo "esqueleto" de treino, implementado em
`src/training/`:

```
1. construir_dataset()     → gera exemplos rotulados (normal / anômalo)
2. train_test_split()      → separa 80% treino / 20% teste, com estratificação
3. RandomForestClassifier  → treina o modelo (200 árvores, profundidade limitada)
4. avaliar_modelo()        → acurácia, precisão, recall, F1, matriz de confusão
5. salvar_modelo_treinado() → grava models/<nome>.joblib + models/<nome>_metricas.json
```

Escolhemos **RandomForestClassifier** (scikit-learn) para os três, de
propósito: é um modelo simples de explicar numa apresentação (é literalmente
uma votação entre várias árvores de decisão), não exige normalização de
atributos, é rápido de treinar sem GPU, e — importante para uso clínico —
permite inspecionar `feature_importances_`, ou seja, dá pra responder "o que
mais pesou nessa decisão?", não só "qual foi a decisão".

### 4.1 Modelo de sinais vitais (`vitais_rf`)

- **Conjunto de dados** (`src/training/dataset_vitais.py`): 1.000 janelas de 15
  minutos de sinais vitais (FC, SpO2, pressão sistólica/diastólica), metade
  normais e metade com uma anomalia injetada (dessaturação, taquicardia,
  hipotensão ou hipertensão — sorteada aleatoriamente). Cada janela vira um
  vetor de 16 atributos: média, desvio-padrão, mínimo e máximo de cada um
  dos 4 sinais.
- **Resultado no conjunto de teste** (200 janelas, nunca vistas no treino):
  **acurácia 100%**, precisão 100%, recall 100%, F1 100%, matriz de
  confusão `[[100, 0], [0, 100]]` (ver `models/vitais_rf_metricas.json`).
- **Features mais importantes**: desvio-padrão da pressão sistólica, desvio
  da SpO2 e desvio da FC — faz sentido: uma anomalia muda a *variação* do
  sinal dentro da janela, não só sua média.

### 4.2 Modelo de movimento (`movimento_rf`)

- **Conjunto de dados** (`src/training/dataset_movimento.py`): 1.000 janelas de 15
  frames de postura (33 pontos-chave do corpo, formato MediaPipe), metade
  paradas/com balanço natural e metade com uma anomalia postural injetada —
  sorteada entre 5 tipos: assimetria de ombro, assimetria de quadril, tronco
  inclinado, joelho travado, ou movimento brusco do corpo inteiro. Cada
  janela é resumida pelas mesmas 8 atributos clínicas de `pose_features.py`
  (assimetria/ângulos/inclinação/velocidade), em média e desvio-padrão — 16
  atributos ao todo (`pose_features.agregar_janela`, reaproveitado tanto no
  treino quanto na inferência, para os dois nunca ficarem fora de sincronia).
- **Resultado no conjunto de teste**: **acurácia 100%**, precisão 100%,
  recall 100%, F1 100% (ver `models/movimento_rf_metricas.json`).
- **Features mais importantes**: desvio-padrão e média da velocidade de
  movimento, seguidas pelo desvio do ângulo de cotovelo direito — um evento
  brusco se manifesta primeiro como pico de velocidade, mas as atributos de
  ângulo/assimetria ajudam o modelo a diferenciar *que tipo* de anomalia
  postural ocorreu (queda vs. rigidez vs. assimetria), não só "houve
  movimento fora do padrão".
- **Uso com vídeo real**: `src/video/pose_analyzer.py` extrai os pontos-chave
  completos via MediaPipe; `src/anomalies/movement_anomaly.py` desliza a
  mesma janela de 15 frames sobre essa sequência real, aplica o modelo
  treinado e devolve, em cada evento, os `motivos` (regra explicável) que
  acompanham a decisão — é o caminho exercitado quando um vídeo é enviado
  pelo frontend ou por `--video`.
- **Modelo alternativo com vídeos reais**: ver seção 4.5.

### 4.3 Modelo de voz (`audio_rf`)

- **Conjunto de dados** (`src/training/dataset_audio.py`): 1.000 exemplos simulados
  diretamente no espaço de características (variabilidade de pitch,
  proporção de silêncio, energia média), com distribuições diferentes para
  "voz normal" e "voz com indício de alteração" (mais monótona, mais
  pausada, mais fraca).
- **Resultado no conjunto de teste**: **acurácia 100%**, precisão 100%,
  recall 100%, F1 100% (ver `models/audio_rf_metricas.json`).
- **Features mais importantes**: as três pesam de forma parecida (~0,30-0,35
  cada) — nenhuma característica isolada domina a decisão, o que é
  esperado, já que as três foram desenhadas para variar juntas entre as
  duas classes.

### 4.4 Eventos sonoros com YAMNet — usando o AudioSet de verdade

O AudioSet ([research.google.com/audioset](https://research.google.com/audioset/conjunto de dados/index.html))
não distribui os arquivos de áudio: fornece só uma planilha de IDs de
vídeos do YouTube com rótulos de *classe de som* (ex: "tosse", "fala"),
então não dá para baixá-lo e treinar um classificador do zero com ele
diretamente (e ainda exigiria baixar cada vídeo do YouTube individualmente).

A alternativa adotada foi usar o **YAMNet**
(`src/audio/audioset_eventos.py`): uma rede já treinada pelo Google
diretamente no AudioSet completo (2 milhões de clipes, 521 classes de som)
e distribuída publicamente via TensorFlow Hub. Isso significa que, embora
não tenhamos treinado nada com o AudioSet neste projeto, o áudio do
paciente É analisado por um modelo que genuinamente aprendeu a
reconhecer sons a partir do AudioSet — só que através do modelo pronto, em
vez do conjunto de dados bruto. No pipeline (`processar_audio_consulta`), o YAMNet
roda em paralelo à transcrição Azure e ao classificador de fadiga/voz,
apontando as classes de som mais prováveis do áudio (ex: tosse, engasgo,
choro, gemido) e marcando quais delas têm relevância clínica direta; a
fusão multimodal (`src/fusion/multimodal_fusion.py`) eleva o risco quando
um evento sonoro considerado grave (engasgo, vômito, grito, gemido) aparece
entre os mais prováveis. Além da classificação nas 521 classes fixas, o
YAMNet também expõe seu embedding interno (`extrair_embedding`), usado como
feature de transfer learning pelo modelo alternativo de voz da seção 4.5.3.

### 4.5 Modelos alternativos treinados com dados reais

Além dos três modelos supervisionados (seções 4.1-4.3, treinados com dados
sintéticos e usados por padrão em produção), o projeto inclui três scripts
de treino **alternativos**, cada um usando dado real de um jeito que não
exige rótulo manual de anomalia:

**4.5.1 Sinais vitais com sinal real (`vitais_isolationforest_physionet`)**

`src/training/treinar_modelo_vitais_physionet.py` treina um `IsolationForest`
**não supervisionado** direto sobre um sinal real: a frequência cardíaca é
derivada das anotações reais de batimento do **MIT-BIH Arrhythmia Database**
(PhysioNet, base aberta, sem necessidade de credencial), via a biblioteca
`wfdb`. Pressão arterial e SpO2 auxiliares são simuladas de forma
correlacionada com essa FC real, só para completar as 4 métricas que o
resto do sistema espera — isso fica documentado no próprio script, sem
esconder o que é real e o que é derivado.

**4.5.2 Movimento com vídeos reais (`movimento_isolationforest_real`)**

`src/training/treinar_modelo_movimento_real.py` processa vídeos reais de
postura considerada normal (colocados pelo usuário em
`amostras/videos_normais/`), extrai os mesmos 16 atributos de janela usados
no treino sintético (seção 4.2) e treina um `IsolationForest` que aprende
como é o padrão normal de movimento — sem precisar de nenhum vídeo real
com queda/espasmo rotulado, que seria difícil de obter eticamente. Um
teste automatizado (`tests/test_dataset_movimento_real.py`) garante que o
conjunto de colunas desse treino real nunca diverge do treino sintético.

**4.5.3 Voz com áudios reais e embeddings do YAMNet (`audio_rf_embeddings_real`)**

`src/training/treinar_modelo_audio_real.py` usa **transfer learning** de
verdade: para cada áudio real organizado em `amostras/audio_por_classe/
<classe>/`, extrai o embedding interno do YAMNet (`src/audio/
audioset_eventos.py::extrair_embedding`, 1024 dimensões) — um resumo
numérico do som aprendido sobre os 2 milhões de clipes do AudioSet — e
treina um `RandomForestClassifier` **supervisionado** por cima desses
embeddings. Diferente do `audio_rf` da seção 4.3 (que simula 3 atributos
diretamente), este modelo aprende a partir do que o YAMNet já sabe sobre
som em geral, precisando de poucos áudios próprios por classe para
funcionar bem.

Por que treinar de um jeito diferente em cada caso? Porque
`IsolationForest` não exige rótulo — então, ao contrário do RandomForest
supervisionado, ele consegue treinar em cima de dado genuinamente real (FC
real, vídeo real) sem precisar injetar uma anomalia artificial para ter o
que aprender. Já o classificador de áudio real continua supervisionado,
mas usando embeddings de transfer learning em vez de rótulo manual de
"anomalia" — a classe (ex: "normal" vs. "tosse") é o próprio nome da pasta
onde o áudio foi colocado, não uma anomalia inferida. A troca de sempre em
aprendizado não supervisionado se aplica aos dois primeiros: sem rótulo de
anomalia, não dá para calcular acurácia/precisão/recall da forma
tradicional — só é possível reportar quantas amostras o próprio modelo
considerou fora do padrão dentro da série usada no treino.

### 4.6 Por que os modelos supervisionados usam dados sintéticos

Os três modelos supervisionados da seção 4.1 a 4.3 (`vitais_rf`,
`movimento_rf`, `audio_rf`) são treinados com dados sintéticos, não com
VitalDB. O motivo é técnico, não de conveniência: treino supervisionado
exige rótulo (`isto é uma anomalia` / `isto não é`), e o VitalDB
([physionet.org/content/vitaldb](https://physionet.org/content/vitaldb/1.0.0/)),
apesar de trazer sinais vitais reais de cirurgias reais, não vem anotado
com "aqui houve uma anomalia" — isso exigiria um médico revisando caso a
caso, fora do escopo desta entrega.

A solução adotada — comum em detecção de anomalias quando não há rótulos
reais — é **injetar a anomalia por código** ao gerar o dado de treino (então
o rótulo nasce sempre correto) e usar o dado real para **validar**
qualitativamente o modelo já treinado. Na prática:

1. Treina-se com dados sintéticos rotulados (seções 4.1 a 4.3).
2. Roda-se o modelo treinado sobre um **caso real do VitalDB**
   (`src/dados/carregador_vitaldb.py`, também disponível na aba "Sinais Vitais"
   do frontend) e observa-se se os episódios sinalizados são
   fisiologicamente plausíveis (ex: uma queda de pressão logo após indução
   anestésica é um evento real e esperado em cirurgia — se o modelo
   sinalizar algo nessa região, é um bom sinal de que ele generaliza).
3. Da mesma forma, o modelo de movimento é validado processando um **vídeo
   real enviado pelo usuário** (aba "Vídeo" do frontend): a trajetória
   extraída pelo MediaPipe vem de um vídeo de verdade, só o *treino* da
   classificação usou trajetórias simuladas.

Essa separação (treino sintético / validação real) é explicitada em cada
tela do frontend e nos logs do pipeline, para não passar a impressão de que
o modelo "viu" dados reais durante o treino.

**Nota sobre acurácia de 100%**: as anomalias sintéticas são injetadas com
uma magnitude grande o suficiente para ficarem claramente separáveis das
janelas normais — por desenho, para que o pipeline de treino (e os testes
automatizados) sejam determinísticos e fáceis de verificar. Isso comprova
que o *código* de treino funciona corretamente; não deve ser lido como "o
modelo é perfeito em dados clínicos reais", que teriam mais ruído e casos
de fronteira. Ver seção 9 (próximos passos) para como isso evoluiria.

## 5. Dados utilizados na inferência/validação

- **Sinais vitais reais**: fonte primária é o **VitalDB**
  (`src/dados/carregador_vitaldb.py`, via biblioteca `vitaldb`), com frequência
  cardíaca, SpO2 e pressão arterial (sistólica/diastólica) reais de um caso
  de cirurgia. Fallback secundário: conjunto de dados aberto **BIDMC** (PhysioNet),
  com FC e SpO2 reais de UTI (`src/dados/carregador_physionet.py`) — nesse
  fallback, a pressão é estimada por correlação simples com a FC, pois o
  BIDMC não traz pressão contínua nos arquivos abertos.
- **CSV externo**: `src/dados/carregador_csv.py` lê qualquer CSV de sinais vitais
  (colunas em português ou inglês) e é a forma mais direta de plugar um
  dado que já existe numa planilha do hospital, sem precisar de API nem
  gerar nada. `amostras/sinais_vitais_exemplo.csv` — um exemplo de 6 linhas com uma
  anomalia clara de FC/SpO2/pressão na quarta linha — vem incluído no
  repositório e é usado nos testes automatizados (`tests/test_csv_loader.py`).
- **Fallback sintético**: quando não há acesso à internet a nenhuma das
  duas fontes acima (por exemplo, em ambientes de rede restrita), ou para
  rodar os testes automatizados sem depender de rede,
  `sample_data_generator.py` gera sinais vitais, prescrições, trajetórias
  de pose e áudio sintéticos — incluindo anomalias propositais, para
  validar os detectores de ponta a ponta em qualquer ambiente.
- **Áudio**: o pipeline aceita qualquer arquivo `.wav`/`.mp3` real de
  consulta, via `--audio` ou pelo frontend. A classificação de eventos
  sonoros (YAMNet) usa o AudioSet indiretamente, através do modelo
  pré-treinado — ver seção 4.4.
- **Vídeo**: o pipeline aceita qualquer vídeo `.mp4`/`.avi`/`.mov` real via
  `--video` ou pelo frontend, inclusive gerando uma versão anotada (seção
  3.1). Não há, até onde levantamos, um conjunto de dados aberto de vídeos de
  fisioterapia/cirurgia pronto para download.

## 6. Resultados obtidos (execução de exemplo)

Rodando `python -m src.main --demo --paciente paciente-042` sobre 12h de
dados sintéticos de sinais vitais (com o modelo `vitais_rf` já treinado
carregado do disco), com duas anomalias propositais inseridas no gerador de
dados:

- **Anomalia 1 (dessaturação)**: SpO2 reduzido artificialmente por volta de
  08:00. **Detectado**: episódio de `spo2` (limite clínico) entre 08:00 e
  08:09, e episódio de `modelo_treinado_vitais` entre 08:00 e 08:22 (22 min,
  23 ocorrências, severidade alta) — o modelo treinado inclusive capturou
  uma janela mais ampla ao redor do evento do que a regra de limite fixo.
- **Anomalia 2 (taquicardia + pico de pressão)**: FC e pressão sistólica
  elevadas por volta de 16:00. **Detectado**: episódio de
  `modelo_treinado_vitais` entre 16:00 e 16:28 (28 min, 29 ocorrências,
  severidade alta), mais os episódios pontuais de limite clínico de FC e
  pressão sistólica no mesmo intervalo.
- O histórico sintético de prescrições incluiu um aumento de dose de
  Dipirona de 500mg para 2000mg (300%) em um único dia — **detectado**
  corretamente como `variacao_dose`, severidade alta.
- **Score de risco final**: 30 pontos → classificado como **ALTO**, com 10
  episódios/alertas de sinais vitais + 1 alerta de prescrição registrados
  no `GerenciadorDeAlertas` (10 de severidade alta, 1 de severidade média).
- **Movimento (modelo treinado)**: rodando sobre uma sequência sintética de
  150 frames de postura com uma queda de ombro/inclinação de tronco
  injetada no meio, o modelo `movimento_rf` sinalizou as janelas ao redor do
  evento, cada uma já com os `motivos` explicáveis (ex: "assimetria de
  ombros") — não só "houve uma anomalia", mas "por quê".

Rodando `python -m src.main --csv amostras/sinais_vitais_exemplo.csv` sobre o CSV de
exemplo (6 amostras, 1 por minuto): a quarta linha (FC 145 bpm, SpO2 89%,
PA 162/101 mmHg) é detectada corretamente como 4 episódios de limite
clínico simultâneos (um por métrica), todos de severidade alta — o modelo
treinado por janela (15 min) não chega a atuar aqui por o arquivo ser mais
curto que uma janela, então quem detecta é a camada de limites clínicos,
como esperado para uma série tão curta.

Os testes automatizados (`pytest tests/ -v`, 27 testes) validam esse
comportamento de forma determinística: construção dos conjuntos de dados de treino,
separabilidade das anomalias sintéticas, detecção das anomalias propositais
via modelo treinado, ausência de falsos positivos em dados estáveis, o
cálculo das atributos clínicas de pose (assimetria/ângulos/motivos), a
consistência de esquema entre o treino sintético e o treino com vídeos
reais, a soma correta de risco na fusão multimodal (incluindo os eventos do
AudioSet) e a extração dos frames anômalos usada na geração do vídeo
anotado.

## 7. Frontend de demonstração

`frontend/app.py` (Streamlit) organiza o fluxo como um atendimento: a
barra lateral cadastra o paciente e mostra o status de cada análise e de
cada modelo treinado. Seis abas compõem o resto da tela:

- **Visão Geral**: junta os resultados de todas as abas já analisadas pela
  mesma função de fusão multimodal usada em `src/main.py`
  (`calcular_risco_paciente`), mostra o nível de risco final e os motivos,
  e gera o relatório consolidado (JSON + Markdown) com botão de download —
  é o "fluxo final do alerta à equipe médica" pedido no desafio, de forma
  interativa.
- **Sinais Vitais / Vídeo / Áudio / Prescrições**: uma aba por modalidade,
  cada uma rodando o pipeline real (dados sintéticos ou reais, conforme a
  seção 5). A aba Vídeo exibe a versão anotada (esqueleto + caixas +
  alertas) quando gerada, com os `motivos` de cada evento de movimento
  anômalo na tabela; a aba Áudio mostra também os eventos do YAMNet.
- **Treinamento**: treina qualquer um dos três modelos principais ao vivo
  (conjunto de dados → treino → métricas → matriz de confusão na tela), mais os três
  modelos alternativos com dados reais (seção 4.5): sinal real do PhysioNet,
  vídeos reais de postura normal e áudios reais com embeddings do YAMNet.

Também é possível rodar o frontend via Docker (`Dockerfile` +
`docker-compose.yml`), útil para garantir que a demonstração funcione de
forma reprodutível independente do ambiente de quem estiver assistindo.

Um roteiro sugerido para o vídeo de demonstração de até 15 minutos está em
[`docs/roteiro_video.md`](roteiro_video.md).

## 8. Como reproduzir

Ver [`README.md`](../README.md) para instruções completas de instalação,
treino dos modelos e execução (linha de comando, frontend ou Docker).

## 9. Próximos passos (fora do escopo desta entrega)

- Anotar (com apoio clínico) uma amostra real do VitalDB com rótulos de
  anomalia, para re-treinar e validar quantitativamente os modelos com
  dados reais, não só sintéticos — os modelos alternativos da seção 4.5 já
  treinam sobre dado real, mas de forma não supervisionada (sem rótulo de
  anomalia) ou usando a própria pasta como rótulo de classe, não de "isto é
  uma anomalia clínica confirmada".
- Treinar um YOLOv8 específico para instrumentos/objetos cirúrgicos, com
  conjunto de dados médico rotulado (hoje o vídeo anotado usa o YOLOv8 genérico,
  pré-treinado em COCO).
- Validar os indicadores acústicos de fadiga/disartria, e os eventos do
  YAMNet, contra uma base de fala patológica real (ex: bases de
  Parkinson/ELA usadas em pesquisa) — o script da seção 4.5.3 já permite
  treinar com áudio real próprio, mas ainda não foi validado contra uma
  base clínica publicada.
- Persistir o histórico de risco por paciente (hoje cada execução/sessão é
  independente) para permitir tendência ao longo da internação.
- Substituir o canal de alerta (hoje log estruturado) por integração real
  com o sistema de plantão do hospital.

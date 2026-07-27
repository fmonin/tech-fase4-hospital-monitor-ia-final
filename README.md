# Monitoramento Multimodal de Pacientes com IA

Projeto da Fase 4 do Tech Challenge da FIAP.

Eu fiz este projeto para mostrar, de um jeito simples, como a inteligência artificial pode ajudar no monitoramento de pacientes. A proposta aqui é juntar três tipos de informação que, sozinhos, já dizem bastante coisa, mas juntos contam uma história bem mais completa sobre o paciente:

- vídeo, para observar postura e objetos na cena
- áudio, para transcrever a fala e analisar o que foi dito
- sinais vitais, para acompanhar frequência cardíaca, SpO2 e pressão

Depois de analisar cada parte separadamente, o sistema junta tudo e gera uma visão geral do risco do paciente. Também tem uma interface em Streamlit para testar tudo na prática, visualizar os resultados e entender como a fusão dos dados acontece.

## O coração do projeto

O centro de tudo aqui é a fusão multimodal. Eu vejo isso como o "coração" do sistema porque é ela que pega as saídas de vídeo, áudio e sinais vitais e transforma isso em uma decisão mais completa.

Sem essa parte central, o projeto seria só um conjunto de analisadores separados. Com a fusão, o sistema consegue responder perguntas mais úteis, como:

- qual modalidade mostrou o maior risco
- se o problema parece mais ligado a movimento, fala ou sinais vitais
- qual foi a pontuação geral do paciente
- quais alertas precisam aparecer primeiro para a equipe

Na prática, o fluxo é assim:

1. cada modalidade gera seus próprios sinais
2. esses sinais são interpretados por regras e modelos treinados
3. a fusão multimodal cruza tudo
4. o relatório final mostra o que mais pesa no risco

Isso é importante porque, em saúde, olhar só uma fonte de dado pode esconder o quadro real do paciente.

## O que o projeto faz

De forma bem resumida, o sistema consegue:

- analisar sinais vitais com dados sintéticos, CSV próprio ou dados reais opcionais
- processar vídeos e identificar postura, objetos e possíveis anomalias de movimento
- processar áudio com Azure Speech to Text e Azure Text Analytics
- identificar eventos sonoros no áudio com YAMNet
- gerar um relatório final com pontuação de risco e principais motivos
- treinar os modelos pela interface ou pela linha de comando

### O que cada parte faz

- **Vídeo**: observa a postura do paciente, procura movimentos estranhos e detecta objetos ou pessoas na cena. Isso ajuda a perceber quando algo fugiu do padrão esperado.
- **Áudio**: pega o som da consulta, transforma em texto, analisa sentimento e procura palavras importantes. Também identifica sons como tosse, engasgo ou respiração diferente.
- **Sinais vitais**: acompanha medidas clínicas básicas e procura variações fora do esperado. Aqui entram frequência cardíaca, SpO2 e pressão.
- **Fusão multimodal**: junta tudo e decide o risco geral. Essa etapa é a que dá sentido ao projeto inteiro, porque evita que o sistema olhe só uma parte da história.
- **Relatório final**: traduz a análise técnica para algo mais fácil de ler, mostrando os principais motivos do alerta.

## Como o fluxo funciona

O projeto usa uma ideia chamada fusão tardia. Isso quer dizer que cada tipo de dado é analisado primeiro por um módulo próprio, e só depois os resultados são combinados. Eu escolhi esse caminho porque ele é mais fácil de explicar, mais fácil de testar e mais próximo de uma solução que consegue crescer com o tempo.

Em linguagem mais simples:

1. o vídeo entra em um pipeline de visão computacional
2. o áudio é transcrito e analisado por serviços e modelos de IA
3. os sinais vitais passam por regras clínicas e por um modelo treinado
4. tudo isso é juntado na fusão multimodal
5. o sistema gera alertas e um relatório final

Eu gostei dessa abordagem porque cada parte fica mais fácil de testar e de explicar.

## Fluxo da arquitetura

![Fluxo da arquitetura multimodal sem prescrições](docs/fluxo_arquitetura.png)

Esse desenho mostra o caminho principal do projeto. O ponto importante aqui é que vídeo, áudio e sinais vitais entram separados, cada um faz sua própria análise, e só no final tudo se junta para formar a decisão final. É justamente esse desenho que ajuda o sistema a não depender de uma única fonte de informação.

## Relatório técnico

O relatório técnico completo está em [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md). Nele eu detalhei:

- como funciona a fusão multimodal entre vídeo, áudio e sinais vitais
- quais modelos e técnicas são aplicados em cada tipo de dado
- quais métricas estão disponíveis nos modelos treinados
- exemplos de anomalias detectadas em sinais vitais, movimento e áudio
- como o vídeo anotado mostra o esqueleto, as caixas de detecção e os alertas de movimento

## Estrutura do projeto

```text
src/
  video/       análise de postura, objetos e vídeo anotado
  audio/       transcrição, sentimento, eventos sonoros e features acústicas
  anomalies/   detectores de anomalias e alertas
  fusion/      fusão multimodal e cálculo do risco final
  dados/       datasets, carregadores e geração de amostras
  reports/     geração do relatório final
  training/    scripts de treino dos modelos
  main.py      execução pela linha de comando
frontend/
  app.py       interface em Streamlit
models/        modelos já treinados e métricas
tests/         testes automatizados
docs/          relatório técnico, roteiro do vídeo e diagrama da arquitetura
amostras/      arquivos de exemplo usados nos testes e na demo
```

## O que já vem pronto no repositório

Eu deixei alguns arquivos prontos para facilitar a demo e também para não depender de nada externo logo no começo:

- modelos treinados em `models/`
- vídeos sintéticos em `amostras/videos_normais/`
- áudio sintético em `amostras/audio_por_classe/`
- um CSV de exemplo em `amostras/sinais_vitais_exemplo.csv`
- um vídeo com anomalia em `amostras/video_exemplo_com_anomalia.mp4`

Isso faz o projeto rodar de ponta a ponta sem precisar baixar nada extra, o que ajuda bastante na apresentação e nos testes locais.

## Pré-requisitos

- Python 3.10 ou 3.11
- pip
- opcional: Docker e Docker Compose
- opcional: conta no Azure, se quiser usar Speech to Text e Text Analytics
- opcional: conta no Kaggle, se quiser baixar o dataset real de fisioterapia

## Instalação

```bash
git clone <url-do-repositorio>
cd hospital-monitor-ia

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Depois disso o projeto já deve abrir normalmente, usando os modelos e arquivos que já estão no repositório.

## Como rodar

### 1. Frontend em Streamlit

```bash
streamlit run frontend/app.py
```

Se preferir:

```bash
python -m streamlit run frontend/app.py
```

O frontend abre em `http://localhost:8501` e mostra, de forma bem visual:

- cadastro do atendimento
- análise de sinais vitais
- análise de vídeo
- análise de áudio
- visão geral com relatório final
- área de treinamento dos modelos

### 2. Docker

```bash
cp .env.example .env
docker compose up --build
```

### 3. Linha de comando

```bash
python -m src.main --demo
python -m src.main --demo --dados-reais
python -m src.main --csv amostras/sinais_vitais_exemplo.csv
python -m src.main --demo --video amostras/videos_normais/normal_1.mp4
python -m src.main --demo --audio amostras/audio_por_classe/normal/normal_1.wav
```

Cada execução gera um relatório em `reports/<id>_<timestamp>.json` e `reports/<id>_<timestamp>.md`.

## Sinais vitais com CSV

Se você quiser testar só os sinais vitais, pode usar este arquivo:

```text
amostras/sinais_vitais_exemplo.csv
```

Ele já vem com uma anomalia clara para mostrar o funcionamento do sistema.

O carregador de CSV aceita colunas em português ou inglês, como:

- `heart_rate` ou `frequencia_cardiaca`
- `spo2`
- `systolic_bp` ou `pressao_sistolica`
- `diastolic_bp` ou `pressao_diastolica`
- `timestamp`

## Treinando os modelos

Os modelos já vêm prontos em `models/`, então não é obrigatório treinar antes de usar.

Mesmo assim, treinar é uma das partes mais importantes do projeto. Foi isso que deu forma para o comportamento do sistema. Sem treino, eu teria só regras soltas. Com treino, o sistema aprende padrões e consegue comparar um caso novo com o que já viu antes.

Treinar bem faz diferença porque:

- melhora a detecção de anomalias
- reduz falso alerta
- deixa o sistema mais consistente entre testes e uso real
- permite avaliar métricas como acurácia, precisão, recall e F1
- ajuda a entender quais atributos pesaram mais na decisão

Se quiser treinar do zero:

```bash
python -m src.training.treinar_tudo
```

Também dá para treinar cada modelo separadamente:

```bash
python -m src.training.treinar_modelo_vitais
python -m src.training.treinar_modelo_movimento
python -m src.training.treinar_modelo_audio
```

### Modelos alternativos com dados reais

```bash
python -m src.training.treinar_modelo_vitais_physionet
python -m src.training.treinar_modelo_movimento_real
python -m src.dados.baixar_kaggle_fisioterapia
python -m src.training.treinar_modelo_movimento_real --pasta amostras/videos_kaggle_fisioterapia/raw
python -m src.training.treinar_modelo_audio_real
```

Esses caminhos são opcionais. Eles servem mais para mostrar que o projeto também conversa com dados reais, quando eles estão disponíveis, e que a estrutura foi pensada para aceitar cenários mais próximos de uso real.

### Por que o treino importa tanto

Na minha visão, essa é a parte que mais sustenta o projeto. Se o modelo não for treinado com cuidado, ele pode:

- marcar como risco algo que não é risco
- deixar passar uma anomalia real
- se comportar bem na demonstração, mas mal em outro cenário

Por isso o projeto separa bem três coisas:

1. construção do conjunto de dados
2. treino do modelo
3. avaliação com métricas

Assim dá para enxergar se o modelo aprendeu de verdade ou se só pareceu funcionar.

## Configurando o Azure

O áudio usa dois serviços do Azure:

- Speech to Text
- Text Analytics

Se quiser ativar isso, crie um `.env` com:

```bash
cp .env.example .env
```

Depois preencha:

- `AZURE_SPEECH_KEY`
- `AZURE_SPEECH_REGION`
- `AZURE_TEXT_ANALYTICS_KEY`
- `AZURE_TEXT_ANALYTICS_ENDPOINT`

Se essas variáveis não estiverem configuradas, o projeto ainda funciona. Só vai pular essa parte do áudio, então ele continua útil mesmo sem credenciais de nuvem.

## Relatório final com OpenAI

O sistema também pode gerar um parecer final mais resumido com a OpenAI.

No `.env`, configure:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_FALLBACK_MODEL=gpt-4o-mini
```

Se a chave não existir, o relatório continua sendo gerado, só sem o texto final da OpenAI. Isso foi importante para não travar a demo por causa de configuração faltando.

## Testes

```bash
pytest tests/ -v
```

Os testes cobrem os detectores de anomalia, o treino, a fusão multimodal, o vídeo anotado e o carregamento de CSV. Na prática, eles ajudam a garantir que as partes principais continuam funcionando quando eu mexo no código.

## Limitações que eu deixei claras

Algumas coisas deste projeto são intencionalmente simplificadas, porque o foco foi entregar um protótipo funcional e explicável, e não uma solução clínica completa:

- os modelos principais são treinados com dados sintéticos rotulados, porque isso facilita criar exemplos controlados para treino e teste
- o AudioSet é acessado via YAMNet, não pelos áudios brutos
- usei MediaPipe Pose no lugar do OpenPose, porque é mais leve e fácil de rodar
- os vídeos baixados do Kaggle não são versionados, só o manifesto fica no repositório
- quando o VitalDB não está disponível, o sistema usa fallback para continuar funcionando

## Licença

Projeto acadêmico da Fase 4 do Tech Challenge.
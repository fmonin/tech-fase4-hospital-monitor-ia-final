# Roteiro do vídeo de demonstração (até 15 minutos)

Este roteiro segue exatamente os pontos exigidos no enunciado do Tech
Challenge: "exemplo prático da análise de áudio e vídeo", "detecção e
resposta a anomalias", "integração dos serviços Azure" e "fluxo final do
alerta à equipe médica". Grave com `streamlit run frontend/app.py` aberto
no navegador; use `.env` configurado com suas chaves Azure para mostrar a
integração em nuvem de verdade (sem chaves, a interface mesmo assim
funciona e avisa claramente que a etapa Azure foi pulada).

## 0:00–1:00 — Introdução

Apresente o problema (monitoramento multimodal preventivo em ambiente
hospitalar), o objetivo do projeto e o aviso de uso acadêmico — o sistema
não substitui diagnóstico ou decisão clínica.

## 1:00–3:00 — Arquitetura e treinamento dos modelos

Mostre rapidamente a estrutura do repositório (`src/video`, `src/audio`,
`src/anomalies`, `src/training`) e o diagrama de fluxo multimodal do
`docs/relatorio_tecnico.md`. Abra a aba **Treinamento** do frontend e
treine ao vivo pelo menos um dos três modelos (sugestão: sinais vitais),
mostrando o conjunto de dados sendo construído, a divisão treino/teste e as métricas
finais (acurácia, matriz de confusão) aparecendo na tela — isso comprova
que o treinamento foi executado de verdade, não é um resultado hard-coded.

## 3:00–6:00 — Sinais vitais

Na aba **Sinais Vitais**, rode primeiro com dados de demonstração
(sintéticos, com anomalias propositais) e mostre o gráfico com os episódios
destacados. Em seguida, se houver internet disponível, troque para "Caso
real do VitalDB" e mostre o mesmo modelo treinado sendo aplicado a um sinal
vital real de cirurgia.

## 6:00–9:00 — Áudio e integração Azure

Na aba **Áudio**, envie um arquivo `.wav` curto (pode ser uma gravação sua
mesmo, simulando uma consulta). Mostre: as características acústicas
extraídas, a classificação do modelo treinado (fadiga/alteração de fala),
os eventos sonoros identificados pelo YAMNet (modelo treinado no AudioSet)
e — se o `.env` estiver configurado — a transcrição via Azure Speech to
Text e os termos críticos/sentimento via Azure Text Analytics.

## 9:00–12:00 — Vídeo

Na aba **Vídeo**, envie um `.mp4` curto de alguém se movimentando
(simulando fisioterapia). Marque a opção de gerar vídeo anotado e mostre o
resultado: esqueleto do MediaPipe sobreposto, caixas do YOLOv8 nos
objetos/pessoas detectados, e o aviso vermelho "ALERTA" nos trechos que o
modelo treinado de movimento classificou como fora do padrão.

## 12:00–14:00 — Fusão multimodal e alerta final

Abra a aba **Visão Geral**, clique em "Gerar relatório consolidado" e
mostre a classificação final de risco (baixo/médio/alto), os motivos
listados e o resumo de alertas por severidade — esse é o "fluxo final do
alerta à equipe médica" pedido no desafio. Baixe o relatório JSON gerado.

## 14:00–15:00 — Conclusão

Comente as limitações documentadas (treino com dados sintéticos rotulados,
validação com dados reais; YOLOv8 genérico em vez de re-treinado para
objetos cirúrgicos específicos; heurísticas de voz não validadas
clinicamente) e os próximos passos descritos em
`docs/relatorio_tecnico.md`.

---

**Checklist antes de gravar:**
- [ ] Modelos treinados (`python -m src.training.treinar_tudo`) ou treinar ao vivo na gravação.
- [ ] `.env` configurado com chaves Azure (opcional, mas recomendado para mostrar a integração de verdade).
- [ ] Um arquivo `.wav` curto e um `.mp4` curto prontos para upload.
- [ ] `streamlit run frontend/app.py` rodando e testado antes de gravar.

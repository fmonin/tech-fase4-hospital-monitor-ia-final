# Vídeos de postura normal

Já vêm alguns vídeos de exemplo nesta pasta (gerados por
`python -m src.dados.gerar_amostras_video_audio`) — pessoas sintéticas
desenhadas por código, mas com forma humana reconhecível pelo MediaPipe
Pose, o suficiente para treinar e testar o pipeline de ponta a ponta sem
precisar de nenhum arquivo externo.

Usados por: `python -m src.training.treinar_modelo_movimento_real`

Para usar vídeos reais próprios, é só colocá-los aqui (mesmo formato,
considerados "normais") ou apontar `--pasta` para outra pasta.

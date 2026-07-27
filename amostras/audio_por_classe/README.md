# Áudios organizados por classe

Já vêm alguns áudios de exemplo em `normal/` e `alterado/` (gerados por
`python -m src.dados.gerar_amostras_video_audio`) — tons sintéticos simples,
com parâmetros diferentes por classe, o suficiente para treinar e testar o
pipeline de ponta a ponta sem precisar de nenhum arquivo externo.

Usados por: `python -m src.training.treinar_modelo_audio_real`

Para usar áudios reais próprios, é só colocá-los numa subpasta por classe
(uma subpasta = um rótulo) ou apontar `--pasta` para outra pasta.

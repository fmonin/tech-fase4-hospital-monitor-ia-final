
import sys
from unittest.mock import MagicMock, patch

# Criar o modulo falso para sys.modules antes de importar audio_pipeline
fake_audioset = MagicMock()
fake_audioset.classificar_eventos_sonoros.return_value = []
sys.modules['src.audio.audioset_eventos'] = fake_audioset

with patch('src.audio.audio_pipeline.extrair_indicadores_de_fala') as mock_extrair, \
     patch('src.audio.audio_pipeline.classificar_risco_fala') as mock_classificar, \
     patch('src.audio.audio_pipeline.transcrever_audio') as mock_transcrever:
     
    mock_extrair.return_value = {'duracao_segundos': 1}
    mock_classificar.return_value = None
    mock_transcrever.return_value = ''
    
    from src.audio.audio_pipeline import processar_audio_consulta
    
    resultado = processar_audio_consulta('arquivo.wav', idioma='en-US')
    
    # Validar as chamadas e resultados
    mock_transcrever.assert_called_once_with('arquivo.wav', idioma='en-US')
    analise_texto_is_none = resultado.get('analise_texto') is None
    contem_aviso = any('não reconheceu fala' in aviso for aviso in resultado.get('avisos', []))
    
    print('TRANSCRIPTION_CALLED_OK:', mock_transcrever.call_args[1].get('idioma') == 'en-US')
    print('ANALISE_TEXTO_IS_NONE:', analise_texto_is_none)
    print('AVISOS_CONTAINS_AVISO:', contem_aviso)
    print('RESULTADO:', resultado)

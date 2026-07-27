from src.video.video_pipeline import _extrair_frames_anomalos


def test_extrai_frames_de_eventos_com_janela():
    eventos = [
        {"janela": [10, 12], "severidade": "media"},
        {"janela": [20, 20], "severidade": "alta"},
    ]
    frames = _extrair_frames_anomalos(eventos)
    assert frames == {10, 11, 12, 20}


def test_extrai_frames_de_eventos_com_frame_unico():
    eventos = [{"frame": 5, "severidade": "media"}, {"frame": 9, "severidade": "alta"}]
    frames = _extrair_frames_anomalos(eventos)
    assert frames == {5, 9}


def test_lista_vazia_devolve_conjunto_vazio():
    assert _extrair_frames_anomalos([]) == set()

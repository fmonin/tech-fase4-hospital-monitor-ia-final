import numpy as np

from src.video.pose_features import extrair_atributos_pose, motivos_da_regra


def _pontos_simetricos() -> np.ndarray:
    pontos = np.zeros((33, 3))
    pontos[:, 2] = 0.9
    pontos[11], pontos[12] = [0.4, 0.3, 0.9], [0.6, 0.3, 0.9]   # ombros
    pontos[23], pontos[24] = [0.42, 0.55, 0.9], [0.58, 0.55, 0.9]  # quadril
    pontos[25], pontos[26] = [0.42, 0.75, 0.9], [0.58, 0.75, 0.9]  # joelhos
    pontos[27], pontos[28] = [0.42, 0.95, 0.9], [0.58, 0.95, 0.9]  # tornozelos
    pontos[13], pontos[14] = [0.35, 0.45, 0.9], [0.65, 0.45, 0.9]  # cotovelos
    pontos[15], pontos[16] = [0.33, 0.6, 0.9], [0.67, 0.6, 0.9]    # punhos
    return pontos


def test_postura_simetrica_nao_gera_motivos():
    atributos = extrair_atributos_pose(_pontos_simetricos())
    assert atributos["assimetria_ombros"] == 0.0
    assert motivos_da_regra(atributos) == []


def test_assimetria_de_ombros_e_detectada():
    pontos = _pontos_simetricos()
    pontos[12][1] = 0.45  # ombro direito bem mais baixo que o esquerdo
    atributos = extrair_atributos_pose(pontos)

    assert atributos["assimetria_ombros"] > 0.10
    assert "assimetria de ombros" in motivos_da_regra(atributos)


def test_pontos_ausentes_nao_quebram_extracao():
    atributos = extrair_atributos_pose(None)
    assert all(np.isnan(v) for v in atributos.values())
    assert motivos_da_regra(atributos) == []


def test_velocidade_calculada_entre_dois_frames():
    pontos_1 = _pontos_simetricos()
    pontos_2 = _pontos_simetricos()
    pontos_2[:, 0] += 0.05  # tudo se move um pouco no eixo x

    atributos = extrair_atributos_pose(pontos_2, pontos_anteriores=pontos_1)
    assert atributos["velocidade_movimento"] > 0

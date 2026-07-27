"""
Roda o treino dos três modelos em sequência (vitais, movimento, áudio).

    python -m src.training.treinar_tudo

Útil como atalho de linha de comando e é a mesma função usada pelo botão
"Treinar modelos" do frontend (frontend/app.py).

Também suporta execução direta do arquivo (botão Play do VS Code):

    python src/training/treinar_tudo.py
"""

from pathlib import Path
import sys


# Suporta execução pelo botão Play (arquivo direto), onde a raiz do projeto
# não entra automaticamente no sys.path.
if __package__ is None or __package__ == "":
    raiz_projeto = Path(__file__).resolve().parents[2]
    if str(raiz_projeto) not in sys.path:
        sys.path.insert(0, str(raiz_projeto))


from src.training import treinar_modelo_audio, treinar_modelo_movimento, treinar_modelo_vitais


def treinar_todos_os_modelos() -> dict:
    """Treina os três modelos e devolve um resumo com as métricas de cada um."""
    _, metricas_vitais = treinar_modelo_vitais.treinar()
    _, metricas_movimento = treinar_modelo_movimento.treinar()
    _, metricas_audio = treinar_modelo_audio.treinar()

    return {
        "vitais": metricas_vitais,
        "movimento": metricas_movimento,
        "audio": metricas_audio,
    }


if __name__ == "__main__":
    resumo = treinar_todos_os_modelos()
    print("\n=== Resumo do treinamento ===")
    for nome, metricas in resumo.items():
        print(f"{nome}: acurácia {metricas['acuracia'] * 100:.1f}% | F1 {metricas['f1_score'] * 100:.1f}%")

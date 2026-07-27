"""Gera a imagem docs/fluxo_arquitetura.png usada no README.

Roda com: python docs/gerar_fluxo_arquitetura.py
Requer Pillow e matplotlib (já estão em requirements.txt / requirements-optional.txt).
"""

import math
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

FONT_REGULAR = font_manager.findfont("DejaVu Sans")
FONT_BOLD = font_manager.findfont("DejaVu Sans:weight=bold")
FONT_ITALIC = font_manager.findfont("DejaVu Sans:style=italic")


def load(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


W, H = 2000, 1450
img = Image.new("RGB", (W, H), "#f4f6fb")
draw = ImageDraw.Draw(img)

for cx, cy, r, color in [
    (140, 140, 130, "#eaf1ff"),
    (1880, 120, 150, "#eafbf6"),
    (1900, 1300, 190, "#f6eeff"),
]:
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

NAVY = "#122841"
SLATE = "#33475b"
MUTED = "#69788d"
WHITE = "#ffffff"
SHADOW = "#dde5f0"


def wrap_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def rounded(xy, fill, outline=None, width=2, radius=26):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def card(x, y, w, h, fill, outline, radius=26, offset=7):
    rounded((x + offset, y + offset, x + w + offset, y + h + offset), SHADOW, radius=radius)
    rounded((x, y, x + w, y + h), fill, outline, width=2, radius=radius)


def text_center(x, y, w, h, text, font, fill, spacing=6):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text((x + (w - tw) / 2, y + (h - th) / 2), text, font=font, fill=fill, spacing=spacing, align="center")


def text_left(x, y, text, font, fill, spacing=6):
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing)


def pill(x, y, text, font, bg, fg=WHITE, pad_x=18, pad_y=9):
    tw = draw.textlength(text, font=font)
    th = font.size
    w = tw + pad_x * 2
    h = th + pad_y * 2
    rounded((x, y, x + w, y + h), bg, radius=999)
    text_center(x, y, w, h, text, font, fg, spacing=0)
    return w, h


def arrow(p1, p2, color, width=4, head=14):
    draw.line((p1, p2), fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    a = (p2[0] - head * math.cos(ang - math.pi / 7), p2[1] - head * math.sin(ang - math.pi / 7))
    b = (p2[0] - head * math.cos(ang + math.pi / 7), p2[1] - head * math.sin(ang + math.pi / 7))
    draw.polygon([p2, a, b], fill=color)


title_font = load(FONT_BOLD, 46)
subtitle_font = load(FONT_REGULAR, 23)
pill_font = load(FONT_BOLD, 19)
col_title_font = load(FONT_BOLD, 21)
box_title_font = load(FONT_BOLD, 25)
box_body_font = load(FONT_REGULAR, 20)
footer_font = load(FONT_ITALIC, 19)
fusion_title_font = load(FONT_BOLD, 32)
fusion_body_font = load(FONT_REGULAR, 21)

text_left(90, 56, "Fluxo Multimodal — Monitoramento de Pacientes com IA", title_font, NAVY)
text_left(
    90, 118,
    "Fusão tardia: cada modalidade é processada por um especialista próprio; os resultados só se encontram na fusão final.",
    subtitle_font, MUTED,
)

px = 90
for label, color in [
    ("3 modalidades principais", "#4f7cff"),
    ("Fusão multimodal + alertas", "#22a06b"),
    ("Relatório final", "#7c5cff"),
]:
    pw, ph = pill(px, 168, label, pill_font, color)
    px += pw + 16

columns = [
    dict(
        title="VÍDEO", accent="#4f7cff", fill="#e8f1ff", border="#b7cdfa",
        top="Upload de vídeo (fisioterapia ou cirurgia)",
        mid="MediaPipe Pose (33 pontos) + YOLOv8 (pessoas e objetos) + atributos clínicos explicáveis",
        bot="Modelo treinado movimento_rf (RandomForest) + vídeo anotado com alertas",
    ),
    dict(
        title="ÁUDIO", accent="#0ea5a4", fill="#e6fbfb", border="#a9e4e1",
        top="Upload de áudio de consulta médica",
        mid="Azure Speech to Text + Azure Text Analytics + librosa (acústica) + YAMNet (AudioSet)",
        bot="Modelo treinado audio_rf (RandomForest) + trechos com sentimento destacado",
    ),
    dict(
        title="SINAIS VITAIS", accent="#22a06b", fill="#eaf8ef", border="#b7e2c4",
        top="VitalDB, PhysioNet, CSV próprio ou dados sintéticos",
        mid="Limites clínicos + agregação por episódios + séries temporais minuto a minuto",
        bot="Modelo treinado vitais_rf (RandomForest) + fallback explicável",
    ),
    dict(
        title="REGRAS", accent="#d98c08", fill="#fff4df", border="#f0d69f",
        top="Entrada opcional de contexto clínico",
        mid="Regras simples e transparentes, como variação de dose ou frequência",
        bot="Sem modelo opaco quando a regra já é suficiente",
    ),
]

col_w = 430
gap = 30
start_x = 90
stage1_y, stage1_h = 250, 150
stage2_y, stage2_h = stage1_y + stage1_h + 50, 210
stage3_y, stage3_h = stage2_y + stage2_h + 50, 170

for i, c in enumerate(columns):
    x = start_x + i * (col_w + gap)

    card(x, stage1_y, col_w, stage1_h, WHITE, c["border"])
    pw, ph = pill(x + 20, stage1_y + 18, c["title"], pill_font, c["accent"])
    wrapped_top = wrap_lines(c["top"], col_title_font, col_w - 40)
    text_left(x + 20, stage1_y + 18 + ph + 12, wrapped_top, col_title_font, NAVY)

    card(x, stage2_y, col_w, stage2_h, c["fill"], c["border"])
    wrapped_mid = wrap_lines(c["mid"], box_body_font, col_w - 44)
    text_left(x + 22, stage2_y + 20, wrapped_mid, box_body_font, SLATE, spacing=8)

    card(x, stage3_y, col_w, stage3_h, WHITE, c["border"])
    wrapped_bot = wrap_lines(c["bot"], box_body_font, col_w - 44)
    text_left(x + 22, stage3_y + 20, wrapped_bot, box_body_font, NAVY, spacing=8)

    draw.rounded_rectangle((x, stage1_y, x + 12, stage3_y + stage3_h), radius=6, fill=c["accent"])

    cx = x + col_w / 2
    arrow((cx, stage1_y + stage1_h), (cx, stage2_y - 8), "#93a4b8", 4)
    arrow((cx, stage2_y + stage2_h), (cx, stage3_y - 8), "#93a4b8", 4)

fusion_y, fusion_h = stage3_y + stage3_h + 70, 170
fusion_w = col_w * 4 + gap * 3
fusion_x = start_x
card(fusion_x, fusion_y, fusion_w, fusion_h, "#0f2136", "#0c1c2d", radius=32, offset=9)
text_left(fusion_x + 40, fusion_y + 26, "FUSÃO MULTIMODAL", fusion_title_font, WHITE)
wrapped_fusion = wrap_lines(
    "Combina os resultados de vídeo, áudio, sinais vitais e regras clínicas para calcular a pontuação de risco do paciente.",
    fusion_body_font, fusion_w - 80,
)
text_left(fusion_x + 40, fusion_y + 76, wrapped_fusion, fusion_body_font, "#d7e3f0", spacing=8)

for i, c in enumerate(columns):
    x = start_x + i * (col_w + gap) + col_w / 2
    arrow((x, stage3_y + stage3_h), (x, fusion_y - 8), c["accent"], 3)

alert_y, alert_h = fusion_y + fusion_h + 70, 140
box_w = (fusion_w - 40) / 2

box1_cx = fusion_x + box_w / 2
box2_cx = fusion_x + box_w + 40 + box_w / 2

card(fusion_x, alert_y, box_w, alert_h, "#fff5f5", "#f2c3c8")
text_left(fusion_x + 30, alert_y + 22, "GERENCIADOR DE ALERTAS", box_title_font, "#9c2f40")
wrapped_alert = wrap_lines(
    "Severidade baixa, média ou alta, encaminhada para a equipe médica.", box_body_font, box_w - 60,
)
text_left(fusion_x + 30, alert_y + 62, wrapped_alert, box_body_font, "#7a3540", spacing=6)

card(fusion_x + box_w + 40, alert_y, box_w, alert_h, "#f2f7ff", "#c7d8f2")
text_left(fusion_x + box_w + 70, alert_y + 22, "RELATÓRIO FINAL", box_title_font, "#28486d")
wrapped_report = wrap_lines(
    "Gerado em JSON e Markdown, com os motivos do risco calculado.", box_body_font, box_w - 60,
)
text_left(fusion_x + box_w + 70, alert_y + 62, wrapped_report, box_body_font, "#365577", spacing=6)

arrow((box1_cx, fusion_y + fusion_h), (box1_cx, alert_y - 8), "#6c7f93", 4)
arrow((box2_cx, fusion_y + fusion_h), (box2_cx, alert_y - 8), "#6c7f93", 4)

footer_y = alert_y + alert_h + 46
text_center(
    0, footer_y, W, 30,
    "Sem prescrições na arquitetura: o foco é vídeo, áudio e sinais vitais, com regras clínicas como complemento.",
    footer_font, MUTED,
)

out = Path(__file__).resolve().parent / "fluxo_arquitetura.png"
img.save(out)
print(out)

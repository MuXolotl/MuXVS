"""Полный каталог встроенных претрейнов `Politrees/RVC_resources`.

Источник: `pretrained/v2/<частота>/<набор>/*.pth` в репозитории.
Имена наборов показываются в интерфейсе как есть, без аннотаций.
"""

BASE_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/"
NO_PRETRAIN = "Без претрейна"
DEFAULT_PRETRAIN = "Default"

# Набор -> {частота: (D-файл, G-файл)}; пути относительно pretrained/v2/<частота>/.
PRETRAINS = {
    "Default": {
        "32k": ("Default/f0D32k.pth", "Default/f0G32k.pth"),
        "40k": ("Default/f0D40k.pth", "Default/f0G40k.pth"),
        "48k": ("Default/f0D48k.pth", "Default/f0G48k.pth"),
    },
    "Snowie v3.1": {
        "32k": ("Snowie/D_SnowieV3.1_32k.pth", "Snowie/G_SnowieV3.1_32k.pth"),
        "40k": ("Snowie/D_SnowieV3.1_40k.pth", "Snowie/G_SnowieV3.1_40k.pth"),
        "48k": ("Snowie/D_SnowieV3.1_48k.pth", "Snowie/G_SnowieV3.1_48k.pth"),
    },
    "Snowie v2": {
        "40k": ("Snowie/D_SnowieV2_40k.pth", "Snowie/G_SnowieV2_40k.pth"),
        "48k": ("Snowie/D_SnowieV2_48k.pth", "Snowie/G_SnowieV2_48k.pth"),
    },
    "Snowie": {
        "40k": ("Snowie/D_Snowie_40k.pth", "Snowie/G_Snowie_40k.pth"),
    },
    "Snowie-X-Rin": {
        "40k": ("Snowie/D_Snowie-X-Rin_40k.pth", "Snowie/G_Snowie-X-Rin_40k.pth"),
    },
    "TITAN-Medium": {
        "32k": ("TITAN/D-f032k-TITAN-Medium.pth", "TITAN/G-f032k-TITAN-Medium.pth"),
        "40k": ("TITAN/D-f040k-TITAN-Medium.pth", "TITAN/G-f040k-TITAN-Medium.pth"),
        "48k": ("TITAN/D-f048k-TITAN-Medium.pth", "TITAN/G-f048k-TITAN-Medium.pth"),
    },
    "KLM v4.3 x3": {
        "32k": ("KLM/D_KLM43_X3_32k.pth", "KLM/G_KLM43_X3_32k.pth"),
        "40k": ("KLM/D_KLM43_X3_40k.pth", "KLM/G_KLM43_X3_40k.pth"),
        "48k": ("KLM/D_KLM43_X3_48k.pth", "KLM/G_KLM43_X3_48k.pth"),
    },
    "KLM v4.3 x2": {
        "32k": ("KLM/D_KLM43_x2_32k.pth", "KLM/G_KLM43_x2_32k.pth"),
    },
    "KLM v4.3 x1": {
        "32k": ("KLM/D_KLM43_x1_32k.pth", "KLM/G_KLM43_x1_32k.pth"),
    },
    "KLM v4.2 x10": {
        "32k": ("KLM/D_KLM42_32k_x10.pth", "KLM/G_KLM42_32k_x10.pth"),
    },
    "KLM v4.2 fp32 T3": {
        "40k": ("KLM/D_KLM42_fp32_T3_40k.pth", "KLM/G_KLM42_fp32_T3_40k.pth"),
    },
    "KLM v4.2 fp32 T2": {
        "40k": ("KLM/D_KLM42_fp32_T2_40k.pth", "KLM/G_KLM42_fp32_T2_40k.pth"),
    },
    "KLM v4.2 fp32 T1": {
        "40k": ("KLM/D_KLM42_fp32_T1_40k.pth", "KLM/G_KLM42_fp32_T1_40k.pth"),
    },
    "KLM v4.2 T4": {
        "40k": ("KLM/D_KLM42_T4_40k.pth", "KLM/G_KLM42_T4_40k.pth"),
    },
    "KLM v4.2 T1": {
        "32k": ("KLM/D_KLM42_T1_32k.pth", "KLM/G_KLM42_T1_32k.pth"),
    },
    "KLM v4.1 T2": {
        "32k": ("KLM/D_KLM41_T2_32k.pth", "KLM/G_KLM41_T2_32k.pth"),
    },
    "KLM v4.1": {
        "32k": ("KLM/D_KLM41_32k.pth", "KLM/G_KLM41_32k.pth"),
        "48k": ("KLM/D_KLM41_48k.pth", "KLM/G_KLM41_48k.pth"),
    },
    "KLM v4.0": {
        "32k": ("KLM/D_KLM40_32k.pth", "KLM/G_KLM40_32k.pth"),
        "40k": ("KLM/D_KLM40_40k.pth", "KLM/G_KLM40_40k.pth"),
        "48k": ("KLM/D_KLM40_48k.pth", "KLM/G_KLM40_48k.pth"),
    },
    "Ov2": {
        "32k": ("Ov2/f0Ov2Super32kD.pth", "Ov2/f0Ov2Super32kG.pth"),
        "40k": ("Ov2/f0Ov2Super40kD.pth", "Ov2/f0Ov2Super40kG.pth"),
    },
    "RIN E3": {
        "40k": ("RIN_E/D_RIN_E3.pth", "RIN_E/G_RIN_E3.pth"),
    },
    "GuideVocalPretrain": {
        "48k": (
            "GuideVocalPretrain/D_GuideVocalPretrain.pth",
            "GuideVocalPretrain/G_GuideVocalPretrain.pth",
        ),
    },
    "Anime": {
        "32k": ("Anime/f0D_AnimePreTrain.pth", "Anime/f0G_AnimePreTrain.pth"),
    },
    "DMR v2": {
        "32k": ("DMR/D_DMR-V2.pth", "DMR/G_DMR-V2.pth"),
    },
    "DMR v1": {
        "32k": ("DMR/D_DMR-V1.pth", "DMR/G_DMR-V1.pth"),
    },
    "DMR v0.8": {
        "32k": ("DMR/D_DMR_V0-8.pth", "DMR/G_DMR_V0-8.pth"),
    },
    "DMR v0.5": {
        "32k": ("DMR/D_dmrV0-5.pth", "DMR/G_dmrV0-5.pth"),
    },
    "IMA Robotic": {
        "32k": ("IMA/D_IMA_Robotic.pth", "IMA/G_IMA_Robotic.pth"),
    },
    "ItaIla": {
        "32k": ("ItaIla/ItaIla_32k_D.pth", "ItaIla/ItaIla_32k_G.pth"),
    },
    "Nanashi v1.5": {
        "32k": ("Nanashi/D_nanashi_v1_5.pth", "Nanashi/G_nanashi_v1_5.pth"),
    },
    "Nanashi v1": {
        "32k": ("Nanashi/D_nanashi_v1.pth", "Nanashi/G_nanashi_v1.pth"),
    },
    "Nezox": {
        "32k": ("Nezox/Nezox_32k_D.pth", "Nezox/Nezox_32k_G.pth"),
    },
    "Rigel": {
        "32k": ("Rigel/D_Rigel_32k.pth", "Rigel/G_Rigel_32k.pth"),
    },
    "Singer 2": {
        "32k": ("Singer/f0D_SingerPreTrain2.pth", "Singer/f0G_SingerPreTrain2.pth"),
    },
    "Singer": {
        "32k": ("Singer/f0D_SingerPreTrain.pth", "Singer/f0G_SingerPreTrain.pth"),
    },
    "UK-A": {
        "32k": ("UK/UKA-Pretrain-D.pth", "UK/UKA-Pretrain-G.pth"),
    },
    "UK-R": {
        "32k": ("UK/UKR-Pretrain-D.pth", "UK/UKR-Pretrain-G.pth"),
    },
}


def rate_label(sample_rate) -> str:
    """Частота вида 48000 -> '48k'."""
    return f"{int(sample_rate) // 1000}k"


def pretrain_choices(sample_rate) -> list:
    """Имена наборов для частоты + пункт без претрейна."""
    rate = rate_label(sample_rate)
    names = [name for name, rates in PRETRAINS.items() if rate in rates]
    if not names:
        names = [name for name, rates in PRETRAINS.items() if "48k" in rates]
    return names + [NO_PRETRAIN]


def pretrain_files(name, sample_rate):
    """(D-файл, G-файл) набора для частоты или None."""
    rates = PRETRAINS.get(name)
    if not rates:
        return None
    return rates.get(rate_label(sample_rate))


def pretrain_rates(name) -> tuple:
    """Частоты, для которых существует набор."""
    return tuple(PRETRAINS.get(name, {}))

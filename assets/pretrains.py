"""Полный каталог встроенных претрейнов `Politrees/RVC_resources`.

Источник: `pretrained/v2/<частота>/<набор>/*.pth` в репозитории.
Имена наборов показываются в интерфейсе как есть, без аннотаций.
"""

BASE_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/"
NO_PRETRAIN = "Без претрейна"
DEFAULT_PRETRAIN = "Default"

# Набор -> {частота: (D-файл, G-файл)}; пути относительно pretrained/v2/<частота>/.
PRETRAINS = {
    # Основные претрейны
    "Default": {
        "32k": ("Default/f0D32k.pth", "Default/f0G32k.pth"),
        "40k": ("Default/f0D40k.pth", "Default/f0G40k.pth"),
        "48k": ("Default/f0D48k.pth", "Default/f0G48k.pth"),
    },
    "Snowie": {
        "40k": ("Snowie/D_Snowie_40k.pth", "Snowie/G_Snowie_40k.pth"),
    },
    "Snowie v2": {
        "40k": ("Snowie/D_SnowieV2_40k.pth", "Snowie/G_SnowieV2_40k.pth"),
        "48k": ("Snowie/D_SnowieV2_48k.pth", "Snowie/G_SnowieV2_48k.pth"),
    },
    "Snowie v3.1": {
        "32k": ("Snowie/D_SnowieV3.1_32k.pth", "Snowie/G_SnowieV3.1_32k.pth"),
        "40k": ("Snowie/D_SnowieV3.1_40k.pth", "Snowie/G_SnowieV3.1_40k.pth"),
        "48k": ("Snowie/D_SnowieV3.1_48k.pth", "Snowie/G_SnowieV3.1_48k.pth"),
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
    "KLM v4.9": {
        "32k": ("KLM/D_KLM_HFG_32k.pth", "KLM/G_KLM_HFG_32k.pth"),
        "40k": ("KLM/D_KLM_HFG_40k.pth", "KLM/G_KLM_HFG_40k.pth"),
        "48k": ("KLM/D_KLM_HFG_48k.pth", "KLM/G_KLM_HFG_48k.pth"),
    },
    "KLM K-pop Universe": {
        "32k": ("KLM/D_KLM_KpopUniverse_32k.pth", "KLM/G_KLM_KpopUniverse_32k.pth"),
        "40k": ("KLM/D_KLM_KpopUniverse_40k.pth", "KLM/G_KLM_KpopUniverse_40k.pth"),
        "48k": ("KLM/D_KLM_KpopUniverse_48k.pth", "KLM/G_KLM_KpopUniverse_48k.pth"),
    },
    # Остальные претрейны (в алфовитном порядке)
    "Anime": {
        "32k": ("Anime/f0D_AnimePreTrain.pth", "Anime/f0G_AnimePreTrain.pth"),
    },
    "Aurora": {
        "32k": ("Aurora/D_Aurora_v2.pth", "Aurora/G_Aurora_v2.pth"),
    },
    "DMR v1": {
        "32k": ("DMR/D_DMR-V1.pth", "DMR/G_DMR-V1.pth"),
    },
    "DMR v2": {
        "32k": ("DMR/D_DMR-V2.pth", "DMR/G_DMR-V2.pth"),
    },
    "GuideVocalPretrain": {
        "48k": (
            "GuideVocalPretrain/D_GuideVocalPretrain.pth",
            "GuideVocalPretrain/G_GuideVocalPretrain.pth",
        ),
    },
    "IMA Robotic": {
        "32k": ("IMA/D_IMA_Robotic.pth", "IMA/G_IMA_Robotic.pth"),
    },
    "ItaIla": {
        "32k": ("ItaIla/ItaIla_32k_D.pth", "ItaIla/ItaIla_32k_G.pth"),
    },
    "Nanashi v1": {
        "32k": ("Nanashi/D_nanashi_v1.pth", "Nanashi/G_nanashi_v1.pth"),
    },
    "Nanashi v1.5": {
        "32k": ("Nanashi/D_nanashi_v1_5.pth", "Nanashi/G_nanashi_v1_5.pth"),
    },
    "Nezox": {
        "32k": ("Nezox/Nezox_32k_D.pth", "Nezox/Nezox_32k_G.pth"),
    },
    "Ov2": {
        "32k": ("Ov2/f0Ov2Super32kD.pth", "Ov2/f0Ov2Super32kG.pth"),
        "40k": ("Ov2/f0Ov2Super40kD.pth", "Ov2/f0Ov2Super40kG.pth"),
    },
    "Rigel": {
        "32k": ("Rigel/D_Rigel_32k.pth", "Rigel/G_Rigel_32k.pth"),
    },
    "RIN E3": {
        "40k": ("RIN_E/D_RIN_E3.pth", "RIN_E/G_RIN_E3.pth"),
    },
    "Singer": {
        "32k": ("Singer/f0D_SingerPreTrain.pth", "Singer/f0G_SingerPreTrain.pth"),
    },
    "Singer 2": {
        "32k": ("Singer/f0D_SingerPreTrain2.pth", "Singer/f0G_SingerPreTrain2.pth"),
    },
    "UK-A": {
        "32k": ("UK/UKA-Pretrain-D.pth", "UK/UKA-Pretrain-G.pth"),
    },
    "UK-R": {
        "32k": ("UK/UKR-Pretrain-D.pth", "UK/UKR-Pretrain-G.pth"),
    },
    "VocalCore": {
        "48k": ("VocalCore/D_VocalCore.pth", "VocalCore/G_VocalCore.pth"),
    },
    "VocalCoreTry3b": {
        "48k": ("VocalCore/D_VocalCoreTry3b.pth", "VocalCore/G_VocalCoreTry3b.pth"),
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

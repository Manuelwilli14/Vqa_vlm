"""Builds notebooks/vqa_colab_kaggle.ipynb by embedding the actual
vqa_engine.py / app.py source (read from disk) into %%writefile cells, so
the notebook always matches the repo files it's generated from.
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "notebooks" / "vqa_colab_kaggle.ipynb"


def _cell_id():
    return uuid.uuid4().hex[:8]


def md(*lines):
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def code(src: str):
    lines = src.splitlines(keepends=True)
    return {"cell_type": "code", "id": _cell_id(), "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def writefile_cell(target: str, content: str):
    return code(f"%%writefile {target}\n" + content)


vqa_engine_src = (ROOT / "vqa_engine.py").read_text(encoding="utf-8")
app_src = (ROOT / "app.py").read_text(encoding="utf-8")

cells = [
    md(
        "# VQA — Visual Question Answering avec Qwen2.5-VL",
        "",
        "Notebook prêt à tourner sur **Colab** ou **Kaggle**. Exécute les cellules dans l'ordre.",
        "",
        "**Avant de lancer :**",
        "- Colab : Exécution → Modifier le type d'exécution → GPU (T4).",
        "- Kaggle : Settings (panneau de droite) → Accelerator → GPU T4 x2, **et** Internet → On (pour télécharger le modèle).",
    ),
    md("## 1. Vérifier le GPU"),
    code("!nvidia-smi"),
    md("## 2. Dépendances\n\nTorch est déjà installé (avec CUDA) sur Colab/Kaggle — on ne touche qu'aux autres paquets."),
    code("!pip install -q -U transformers accelerate gradio pillow"),
    md("## 3. Fichiers du projet\n\nMêmes fichiers que le repo GitHub, générés ici pour que le notebook soit autonome."),
    writefile_cell("vqa_engine.py", vqa_engine_src),
    writefile_cell("app.py", app_src),
    md(
        "## 4. Test rapide en ligne de commande",
        "",
        "Un aller-retour sur une image d'exemple avant de lancer l'interface complète — "
        "si ça répond correctement ici, l'UI Gradio fonctionnera pareil.",
    ),
    code(
        "from skimage import data\n"
        "from PIL import Image\n"
        "\n"
        "sample = Image.fromarray(data.astronaut())  # image d'exemple incluse avec scikit-image, pas de téléchargement\n"
        "sample.save('sample.jpg')\n"
        "sample\n"
    ),
    code(
        "from vqa_engine import ask\n"
        "\n"
        "print(ask(sample, \"Décris cette image en une phrase.\", mode=\"Réponse détaillée\"))\n"
    ),
    md(
        "## 5. Lancer l'interface Gradio",
        "",
        "`share=True` donne un lien public temporaire (`*.gradio.live`, valable quelques heures) — "
        "pratique pour tester depuis ton téléphone ou montrer la démo sans rien déployer.",
    ),
    code(
        "from app import demo\n"
        "\n"
        "demo.launch(share=True, debug=True)\n"
    ),
    md(
        "## Pour aller plus loin",
        "",
        "- Change `MODEL_NAME` dans `vqa_engine.py` (cellule 3) pour essayer `Qwen/Qwen2.5-VL-7B-Instruct` si le GPU a assez de VRAM.",
        "- Ajoute un mode de prompt dans `PROMPT_MODES` et compare-le aux 4 existants sur la même image+question.",
        "- Une fois content du résultat, télécharge `vqa_engine.py` / `app.py` (mis à jour si tu les as modifiés ici) pour les pousser sur GitHub à côté de ce notebook.",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "vqa_colab_kaggle.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")

# VQA — Visual Question Answering avec Qwen2.5-VL

Petit outil : tu donnes une image + une question en langage naturel, le
modèle (Qwen2.5-VL-3B-Instruct, vision-language model open source) répond.
Interface Gradio, tourne sur un seul GPU grand public (ou gratuit sur
Colab/Kaggle).

L'intérêt du projet n'est pas d'appeler le modèle une fois — c'est le
**prompt engineering** : un VLM répond différemment selon comment on lui
demande. Ce repo compare 4 stratégies de prompt système sur la même paire
image/question :

| Mode | Ce que ça change |
|---|---|
| **Réponse courte** | Force une réponse en quelques mots, pas d'explication |
| **Réponse détaillée** | Réponse + justification par ce qui est visible dans l'image |
| **Comptage précis** | Liste chaque instance une par une avant de compter (les VLM comptent mal en zero-shot direct — lister d'abord aide beaucoup) |
| **Vérification** | Brouillon de réponse, puis re-vérification explicite avant la réponse finale |

## Structure

```
vqa-vlm-tool/
├── vqa_engine.py   # chargement du modèle + les 4 prompts système + ask()
├── app.py          # interface Gradio
├── requirements.txt
└── notebooks/
    └── vqa_colab_kaggle.ipynb   # prêt à tourner sur Colab/Kaggle (GPU gratuit)
```

## Lancer en local (avec un GPU NVIDIA)

```bash
pip install -r requirements.txt
python app.py
```

Ouvre l'URL locale affichée dans le terminal (par défaut `http://127.0.0.1:7860`).

## Lancer sur Colab / Kaggle

Ouvre `notebooks/vqa_colab_kaggle.ipynb` (Colab : Fichier → Importer un
notebook ; Kaggle : Créer un notebook → importer le fichier), vérifie que
l'accélérateur GPU est activé (Colab : Runtime → Modifier le type
d'exécution → T4 GPU ; Kaggle : Settings → Accelerator → GPU T4 x2), puis
exécute les cellules dans l'ordre. La dernière cellule lance Gradio avec
`share=True`, ce qui donne un lien public temporaire (`*.gradio.live`) —
pratique pour montrer la démo sans rien déployer.

**Kaggle uniquement** : active Settings → Internet → On (nécessaire pour
télécharger le modèle depuis Hugging Face).

## Changer de modèle

`MODEL_NAME` en haut de `vqa_engine.py` :

- `Qwen/Qwen2.5-VL-3B-Instruct` (défaut) — tourne bien sur un T4 16GB.
- `Qwen/Qwen2.5-VL-7B-Instruct` — meilleures réponses, mais plus lent à
  charger et plus gourmand en VRAM (compte ~16GB rien que pour les poids en
  bf16 ; passe par une version quantifiée 4-bit avec `bitsandbytes` si ton
  GPU a moins de mémoire).

## Notes techniques

- Chargement du modèle une seule fois (mis en cache), pas à chaque requête.
- Utilise `apply_chat_template(..., tokenize=True, return_dict=True,
  return_tensors="pt")` pour passer directement une image PIL — pas besoin
  du paquet `qwen_vl_utils`, les versions récentes de `transformers`
  gèrent ça nativement (depuis `transformers>=4.48`).
- `do_sample=False` (génération déterministe / greedy) pour avoir des
  réponses reproductibles d'un run à l'autre sur la même image+question.
- Piste d'extension évidente : ajouter une évaluation automatique (batch de
  questions avec réponses attendues) pour comparer les 4 modes de prompt
  de façon chiffrée plutôt qu'à l'œil.

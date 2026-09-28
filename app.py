"""
app.py — Gradio front-end for the VQA tool.

Run locally:  python app.py
Run on Colab/Kaggle: see notebooks/vqa_colab_kaggle.ipynb (uses share=True
since those platforms don't expose local ports directly).
"""
import gradio as gr
from vqa_engine import ask, PROMPT_MODES, DEFAULT_MODE, MODEL_NAME

with gr.Blocks(title="VQA — Qwen2.5-VL") as demo:
    gr.Markdown(
        f"# 🖼️ Visual Question Answering\n"
        f"Pose une question sur une image en langage naturel.\n\n"
        f"Modèle : `{MODEL_NAME}` (chargé au premier appel, ~10-30s selon le matériel)."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Image")
            question_in = gr.Textbox(
                label="Question",
                placeholder="Combien de personnes portent du rouge ?",
            )
            mode_in = gr.Radio(
                choices=list(PROMPT_MODES.keys()),
                value=DEFAULT_MODE,
                label="Style de réponse",
                info="Change le prompt système envoyé au modèle — pas juste le ton, la stratégie de raisonnement.",
            )
            with gr.Accordion("Options avancées", open=False):
                max_tokens_in = gr.Slider(
                    32, 512, value=300, step=32, label="Longueur max. de la réponse (tokens)"
                )
            submit_btn = gr.Button("Demander", variant="primary")
        with gr.Column():
            answer_out = gr.Textbox(label="Réponse", lines=12)

    submit_btn.click(
        fn=ask,
        inputs=[image_in, question_in, mode_in, max_tokens_in],
        outputs=answer_out,
    )
    question_in.submit(
        fn=ask,
        inputs=[image_in, question_in, mode_in, max_tokens_in],
        outputs=answer_out,
    )

if __name__ == "__main__":
    demo.launch()

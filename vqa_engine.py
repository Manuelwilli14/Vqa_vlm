"""
vqa_engine.py — loads Qwen2.5-VL-Instruct and answers a question about an image.

The interesting part of this project isn't "call the model once" — it's the
prompt engineering: a free-form system prompt handles open questions fine,
but VLMs are notoriously unreliable at counting and at admitting uncertainty,
so those question types get their own tuned instructions below. Swap the
active mode from the Gradio UI (see app.py) to compare them on the same
image + question.
"""
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Bump to "Qwen/Qwen2.5-VL-7B-Instruct" if you have >16GB VRAM — better
# answers, slower load, needs more memory than a free-tier T4 comfortably
# gives alongside everything else.
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---------------------------------------------------------------- prompts ---
# Each "mode" is a system prompt tuned for a different question style.
PROMPT_MODES = {
    "Réponse courte": (
        "You are a precise visual question answering assistant. Look "
        "carefully at the image and answer the user's question in the "
        "fewest words possible — ideally a single word, number, or short "
        "phrase. Do not explain your reasoning. If the answer is not "
        "visible in the image, say \"Je ne sais pas\" instead of guessing."
    ),
    "Réponse détaillée": (
        "You are a visual question answering assistant. Look carefully at "
        "the image and answer the user's question, then briefly justify "
        "your answer by describing what you see that supports it. If "
        "something is uncertain or not visible, say so instead of guessing."
    ),
    "Comptage précis": (
        "You are a careful visual counting assistant. To answer a counting "
        "question: first list every matching instance you can individually "
        "identify, one per line, with a short description of where it is "
        "in the image (for example '1. person, top-left, red jacket'). "
        "Only after listing every instance, give one final line in the "
        "exact form 'Total: N'. If two instances might be the same object "
        "seen twice, count them separately and note the ambiguity."
    ),
    "Vérification": (
        "You are a careful visual question answering assistant. First give "
        "a draft answer to the question. Then re-examine the image "
        "specifically to check that draft for mistakes (wrong count, wrong "
        "color, misread text, etc). Finally give a line in the exact form "
        "'Réponse finale : ...' with your checked answer."
    ),
}

DEFAULT_MODE = "Réponse courte"

_model = None
_processor = None


def load_model(model_name: str = MODEL_NAME):
    """Loads the model once and caches it; safe to call on every request."""
    global _model, _processor
    if _model is None:
        print(f"Loading {model_name} ...")
        _processor = AutoProcessor.from_pretrained(model_name)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _model.eval()
        print("Model loaded.")
    return _model, _processor


def ask(image: Image.Image, question: str, mode: str = DEFAULT_MODE,
        max_new_tokens: int = 300) -> str:
    """Answers `question` about `image` using the system prompt for `mode`."""
    if image is None:
        return "Envoie d'abord une image."
    if not question or not question.strip():
        return "Écris une question."

    model, processor = load_model()
    system_prompt = PROMPT_MODES.get(mode, PROMPT_MODES[DEFAULT_MODE])

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": question},
            ],
        },
    ]

    # transformers >= 4.48 lets apply_chat_template load the image and
    # return ready-to-use tensors directly — no separate vision-preprocessing
    # helper package needed.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

    response = processor.decode(
        output_ids[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


if __name__ == "__main__":
    # Quick CLI smoke test: python vqa_engine.py path/to/image.jpg "question"
    import sys
    if len(sys.argv) < 3:
        print('Usage: python vqa_engine.py image.jpg "your question"')
        raise SystemExit(1)
    img = Image.open(sys.argv[1])
    print(ask(img, sys.argv[2]))

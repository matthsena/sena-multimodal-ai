# messages = [
#     {"role": "user", "content": """
#     Estou utilizando alguns modelos de imagem para extrair caracteristicas de imagens, 
#     vou passar o output em formato de texto, dito isso voce tem que me dizer se as duas 
#     imagens representam a mesma cena.
    
#     Imagem 1: 
#         - Predições com Inception v3 e ResNet-50: soccer ball 78% e rugby ball  58%.
#         - Segmentacao pantonica: 1 sports ball, 9 person, 1 playingfield e 1 fence. 
#         - Textos extraidos com OCR: RVARANE, RVA, Teamvvewer
#     Imagem 2: 
#         - Predições com Inception v3 e ResNet-50: rugby ball 69% e soccer ball 51%.
#         - Segmentacao pantonica: 1 sports ball, 10 person, 1 playingfield.
#         - Textos extraidos com OCR:TeamVirvel, FWaYS, IFAD
#     """},
# ]

from transformers import AutoTokenizer, pipeline
import torch
from collections import Counter

model = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

def merge_and_sum_predictions(inception_preds, resnet50_preds):
    summed_values = Counter(dict(inception_preds))
    summed_values.update(Counter(dict(resnet50_preds)))
    summed_values = {k: round(v * 50, 2) for k, v in summed_values.items()}
    result = sorted(summed_values.items(),
                    key=lambda x: x[1], reverse=True)[:5]
    return [f'{k} ({v}%)' for k, v in result]


def discart_or_format_ocr(ocr_list):
    result = [item for item in ocr_list if item[1] > 0.25]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [f'{item[0]} ({round(item[1], 2)}%)' for item in result]


def format_panoptic_list(panoptic_list):
    counts = Counter(panoptic_list)
    result = [f'{count} {item}' for item, count in counts.items()]
    return result


def generate_text(preds1, preds2, panoptic1, panoptic2, ocr1, ocr2):
    messages = [
        {"role": "user", "content": f"""
         Bellow are the results of the image models in text format, with output and probabilities.

         Image 1:
          - Predictions with Inception v3 and ResNet-50: {preds1}.
          - Panoptic segmentation: {panoptic1}.
          - Texts extracted with OCR: {ocr1}.

         Image 2:
          - Predictions with Inception v3 and ResNet-50: {preds2}.
          - Panoptic segmentation: {panoptic2}.
          - Texts extracted with OCR: {ocr2}.

        tell me if the two images represent the same scene.
        """},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95
    )
    return outputs[0]["generated_text"][len(prompt):]

import os
# OCR
from easyocr_custom import predict as ocr_predictor
# DETECTRON 2
from detectron2_custom import panoptic_predictor, keypoint_predictor
# Inception3
from inception3_custom import predict as inception3_predictor
# Resnet50
from resnet50_custom import predict as resnet50_predictor
# LLM
from gemma_2b_single_img import discart_or_format_ocr, format_panoptic_list, generate_text as gemma_2b_predictor, merge_and_sum_predictions

import gradio as gr
import numpy as np
import cv2


def pil_to_cv2(image):
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def image_process(image):
    results = []
    fallback_image = cv2.imread('./images/fallback.png')

    try:
        inception_preds = inception3_predictor(image)
        inception_predictions = list(
            map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", inception_preds))
        inception_markdown = f"# Inception v3\n### Top 5 Predições:\n{''.join(inception_predictions)}"
        results.append(inception_markdown)
    except Exception as e:
        results.append(f"Error in Inception3: {e}")

    try:
        resnet50_preds = resnet50_predictor(image)
        resnet50_predictions = list(
            map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", resnet50_preds))
        resnet50_markdown = f"# ResNet-50\n### Top 5 Predições:\n{''.join(resnet50_predictions)}"
        results.append(resnet50_markdown)
    except Exception as e:
        results.append(f"Error in Resnet50: {e}")

    image_cv2 = pil_to_cv2(image)

    try:
        panoptic, extracted_classes = panoptic_predictor(image_cv2)
        results.append(pil_to_cv2(panoptic))
        panoptic_predicitions = list(
            map(lambda x: f"<b>{x}</b><br/>", extracted_classes))
        results.append(
            f"# Panoptic Predictor\n### Classes extraídas:\n{''.join(panoptic_predicitions)}")
    except Exception as e:
        results.append(fallback_image)
        results.append(f"Error in Panoptic: {e}")

    try:
        ocr_image, valuesAndProbsOCR = ocr_predictor(image_cv2)
        results.append(pil_to_cv2(ocr_image))
        ocr_predictions = list(
            map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", valuesAndProbsOCR))
        results.append(
            f"# OCR\n### Textos extraídos:\n{''.join(ocr_predictions)}")
    except Exception as e:
        results.append(fallback_image)
        results.append(f"Error in OCR: {e}")


    text = gemma_2b_predictor(merge_and_sum_predictions(inception_preds, resnet50_preds), format_panoptic_list(
        extracted_classes), discart_or_format_ocr(valuesAndProbsOCR))

    results.append(text)

    return results


demo = gr.Interface(
    image_process,
    inputs=[
        gr.Image(type="pil"),
    ],
    outputs=[
        gr.Markdown(),
        gr.Markdown(),
        gr.Image(label="Panoptic Predictor"),
        gr.Markdown(),
        gr.Image(label="OCR"),
        gr.Markdown(),
        gr.Textbox(lines=5, label="LLaMA 2 Output")
    ],
    # flagging_options=["blurry", "incorrect", "other"],
    examples=[
        [os.path.join(os.path.dirname(__file__), "images/game1.jpg")],
        [os.path.join(os.path.dirname(__file__),
                      "images/placabrasil.jpg")],
        [os.path.join(os.path.dirname(__file__), "images/usp.jpg")]
    ],
)

if __name__ == "__main__":
    demo.launch()

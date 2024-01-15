# LLAMA
import os
from llama2_custom import prompt_creator, llama2
# OCR
from easyocr_custom import predict as ocr_predictor
# DETECTRON 2
from detectron2_custom import panoptic_predictor, keypoint_predictor
# Inception3
from inception3_custom import predict as inception3_predictor
# Resnet50
from resnet50_custom import predict as resnet50_predictor

import gradio as gr
import numpy as np
import cv2

from threading import Thread


def pil_to_cv2(image):
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def image_process(image, use_inception, use_resnet50, use_panoptic, use_keypoint, use_ocr):
    results = []
    fallback_image = cv2.imread('./images/fallback.png')

    if use_inception:
        try:
            inception_preds = inception3_predictor(image)
            inception_predictions = list(map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", inception_preds))
            inception_markdown = f"# Inception v3\n### Top 5 Predições:\n{''.join(inception_predictions)}"
            results.append(inception_markdown)
        except Exception as e:
            print(f"Error in Inception v3: {e}")
            results.append("Error in Inception3")

    if use_resnet50:
        try:
            resnet50_preds = resnet50_predictor(image)
            resnet50_predictions = list(map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", resnet50_preds))
            resnet50_markdown = f"# ResNet-50\n### Top 5 Predições:\n{''.join(resnet50_predictions)}"
            results.append(resnet50_markdown)
        except Exception as e:
            print(f"Error in Resnet50: {e}")
            results.append("Error in Resnet50")

    image_cv2 = pil_to_cv2(image)

    if use_panoptic:
        try:
            panoptic, extracted_classes = panoptic_predictor(image_cv2)
            results.append(pil_to_cv2(panoptic))
            panoptic_predicitions = list(map(lambda x: f"<b>{x}</b><br/>", extracted_classes))
            results.append(f"# Panoptic Predictor\n### Classes extraídas:\n{''.join(panoptic_predicitions)}")
        except Exception as e:
            print(f"Error in Panoptic: {e}")
            results.append(fallback_image)
            results.append("Error in Panoptic")
    else: 
        results.append(None)

    if use_keypoint:
        try:
            keypoint = keypoint_predictor(image_cv2)
            results.append(pil_to_cv2(keypoint))
        except Exception as e:
            print(f"Error in Keypoint: {e}")
            results.append(fallback_image)
    else:
        results.append(None)

    if use_ocr:
        try:
            ocr_image, valuesAndProbsOCR = ocr_predictor(image_cv2)
            results.append(pil_to_cv2(ocr_image))
            ocr_predictions = list(map(lambda x: f"<b>{x[0]}</b> - {round(x[1], 2)}<br/>", valuesAndProbsOCR))
            results.append(f"# OCR\n### Textos extraídos:\n{''.join(ocr_predictions)}")
        except Exception as e:
            print(f"Error in OCR: {e}")
            results.append(fallback_image)
            results.append("Error in OCR")
    else:
        results.append(None)

    return results


demo = gr.Interface(
    image_process,
    inputs=[
        gr.Image(type="pil"),
        gr.Checkbox(label="Use Inception3", value=True),
        gr.Checkbox(label="Use ResNet50", value=True),
        gr.Checkbox(label="Use Panoptic Predictor", value=True),
        gr.Checkbox(label="Use Keypoint Predictor", value=True),
        gr.Checkbox(label="Use OCR", value=True)
    ],
    outputs=[
        gr.Markdown(),
        gr.Markdown(),
        gr.Image(label="PANOPTIC_PREDICTOR"),
        gr.Markdown(),
        gr.Image(label="KEYPOINT_PREDICTOR"),
        gr.Image(label="OCR"),
        gr.Markdown(),
    ],
    # flagging_options=["blurry", "incorrect", "other"],
    examples=[
        [os.path.join(os.path.dirname(__file__), "images/game1.jpg"), True, True, True, True, True],
        [os.path.join(os.path.dirname(__file__), "images/placabrasil.jpg"), True, True, True, True, True],
        [os.path.join(os.path.dirname(__file__), "images/usp.jpg"), True, True, True, True, True]
    ],
)

if __name__ == "__main__":
    demo.launch()

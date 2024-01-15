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
            print(f"Inception: {inception_preds}")
        except Exception as e:
            print(f"Error in Inception3: {e}")

    if use_resnet50:
        try:
            resnet50_preds = resnet50_predictor(image)
            print(f"Resnet50: {resnet50_preds}")
        except Exception as e:
            print(f"Error in Resnet50: {e}")

    image_cv2 = pil_to_cv2(image)

    if use_panoptic:
        try:
            panoptic, extracted_classes = panoptic_predictor(image_cv2)
            results.append(pil_to_cv2(panoptic))
            print(f"Extracted Classes: {extracted_classes}")
        except Exception as e:
            print(f"Error in Panoptic: {e}")
            results.append(fallback_image)
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
            print(f"OCR: {valuesAndProbsOCR}")
        except Exception as e:
            print(f"Error in OCR: {e}")
            results.append(fallback_image)
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
        gr.Image(label="PANOPTIC_PREDICTOR"),
        gr.Image(label="KEYPOINT_PREDICTOR"),
        gr.Image(label="OCR"),
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

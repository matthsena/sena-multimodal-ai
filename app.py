# LLAMA
import os
from llama2_custom import prompt_creator, llama2
# OCR
from easyocr_custom import predict as ocr_predictor
# DETECTRON 2
from detectron2_custom import panoptic_predictor, keypoint_predictor

import gradio as gr
import numpy as np
import cv2


def pil_to_cv2(image):
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def image_process(image):
    image = pil_to_cv2(image)
    panoptic, extracted_classes  = panoptic_predictor(image)
    keypoint = keypoint_predictor(image)
    ocr_image, valuesAndProbsOCR = ocr_predictor(image)

    print(f"Extracted Classes: {extracted_classes}")
    print(f"OCR: {valuesAndProbsOCR}")

    return pil_to_cv2(panoptic), pil_to_cv2(keypoint), pil_to_cv2(ocr_image)


demo = gr.Interface(
    image_process,
    gr.Image(type="pil"),
    outputs=[
        gr.Image(label="PANOPTIC_PREDICTOR"),
        gr.Image(label="KEYPOINT_PREDICTOR"),
        gr.Image(label="OCR"),
    ],
    # flagging_options=["blurry", "incorrect", "other"],
    examples=[
        os.path.join(os.path.dirname(__file__), "images/game1.jpg"),
        os.path.join(os.path.dirname(__file__), "images/placabrasil.jpg"),
        os.path.join(os.path.dirname(__file__), "images/usp.jpg")
    ],
)

if __name__ == "__main__":
    demo.launch()

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
from gemma_2b_multi_imgs import discart_or_format_ocr, format_panoptic_list, generate_text as gemma_2b_predictor, merge_and_sum_predictions

import gradio as gr
import numpy as np
import cv2


def pil_to_cv2(image):
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def image_process(image1, image2):
    inception_preds1 = inception3_predictor(image1)
    inception_preds2 = inception3_predictor(image2)

    resnet50_preds1 = resnet50_predictor(image1)
    resnet50_preds2 = resnet50_predictor(image2)

    image_cv2_1 = pil_to_cv2(image1)
    image_cv2_2 = pil_to_cv2(image2)

    _, extracted_classes1 = panoptic_predictor(image_cv2_1)
    _, extracted_classes2 = panoptic_predictor(image_cv2_2)

    _, text_probs1 = ocr_predictor(image_cv2_1)
    _, text_probs2 = ocr_predictor(image_cv2_2)

    preds1 = merge_and_sum_predictions(inception_preds1, resnet50_preds1)
    preds2 = merge_and_sum_predictions(inception_preds2, resnet50_preds2)

    panoptic1 = format_panoptic_list(extracted_classes1)
    panoptic2 = format_panoptic_list(extracted_classes2)

    ocr1 = discart_or_format_ocr(text_probs1)
    ocr2 = discart_or_format_ocr(text_probs2)

    text = gemma_2b_predictor(preds1, preds2, panoptic1, panoptic2, ocr1, ocr2)

    return text


demo = gr.Interface(
    image_process,
    inputs=[
        gr.Image(type="pil"),
        gr.Image(type="pil"),
    ],
    outputs=gr.Textbox(lines=5, label="LLaMA 2 Output"),
    # flagging_options=["blurry", "incorrect", "other"],
    # examples=[
    #     [os.path.join(os.path.dirname(__file__), "images/game1.jpg")],
    #     [os.path.join(os.path.dirname(__file__),
    #                   "images/placabrasil.jpg")],
    #     [os.path.join(os.path.dirname(__file__), "images/usp.jpg")]
    # ],
)

if __name__ == "__main__":
    demo.launch()

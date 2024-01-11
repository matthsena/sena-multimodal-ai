import sys, os

sys.path.insert(0, os.path.abspath('./detectron2'))

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw

# OCR
import easyocr

# DETECTRON 2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json

def pil_to_cv2(image):
  opencv_image = np.array(image)
  opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
  return opencv_image

def panoptic_predictor(image):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
  predictor = DefaultPredictor(cfg)
  panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

  dset_meta = MetadataCatalog.get("coco_2017_train")
  dset_meta_stuffs = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

  # dataset_dicts = load_coco_json("annotations/instances_train2017.json", "train2017")
  # stuff_classes = dataset_dicts["stuff_classes"]
  # print("Stuff classes:", stuff_classes)

  for info in segments_info:
      print(info)
      if info["isthing"] == True:
          cat = dset_meta.thing_classes[info['category_id']]
          print(cat, info['score'])
      else:
          cat = dset_meta_stuffs.stuff_classes[info['category_id']]
          print(cat)

  return pil_to_cv2(out.get_image()[:, :, ::-1])

def keypoint_predictor(image):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
  predictor =  DefaultPredictor(cfg)
  keypoint_seg = predictor(image)
  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(keypoint_seg["instances"].to("cpu"))
  return pil_to_cv2(out.get_image()[:, :, ::-1])

# OCR
def ocr_predictor(image):
    reader = easyocr.Reader(['en', 'pt', 'es', 'fr', 'de', 'it'], gpu=False)
    result = reader.readtext(image)

    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print("text: ", text, "\nprob: ", prob)
    return pil_to_cv2(image)


def image_process(image):
  image = pil_to_cv2(image)
  panoptic = panoptic_predictor(image)
  keypoint = keypoint_predictor(image)
  ocr = ocr_predictor(image)

  return panoptic, keypoint, ocr


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


import sys
import os

sys.path.insert(0, os.path.abspath('./detectron2'))

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


# DETECTRON 2
def panoptic_predictor(image):
  try:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
      "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
      "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
    v = Visualizer(image[:, :, ::-1],
             MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(
      panoptic_seg.to("cpu"), segments_info)

    dset_meta = MetadataCatalog.get("coco_2017_train")
    dset_meta_stuffs = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    extracted_classes = []

    for info in segments_info:
      if info["isthing"] == True:
        if info["score"] < 0.5:
          return
        cat = dset_meta.thing_classes[info['category_id']]
        extracted_classes.append(cat)
      else:
        cat = dset_meta_stuffs.stuff_classes[info['category_id']]
        extracted_classes.append(cat)
    return out.get_image()[:, :, ::-1], extracted_classes
  except Exception as e:
    print(f"An error occurred on panoptic_predictor: {e}")
    return None, None


def keypoint_predictor(image):
  try:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
      "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
      "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    keypoint_seg = predictor(image)
    v = Visualizer(image[:, :, ::-1],
             MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(keypoint_seg["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]
  except Exception as e:
    print(f"An error occurred on keypoint_predictor: {e}")
    return None

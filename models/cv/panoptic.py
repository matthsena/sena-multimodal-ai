from collections import Counter
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation, logging
import json
from torch.amp import autocast

logging.set_verbosity_error()

class PanopticPredictor:
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50-panoptic")
        self.model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic", ignore_mismatched_sizes=True)
        with open('resources/coco_2017_cat.json') as f:
            self.coco_classes = json.load(f)

    def predict(self, image):
        try:
            with torch.no_grad():
                with autocast('cuda'):
                    inputs = self.feature_extractor(
                        images=image, return_tensors="pt")
                    outputs = self.model(**inputs)
                    processed_sizes = torch.as_tensor(
                        inputs["pixel_values"].shape[-2:]).unsqueeze(0)
                    result = self.feature_extractor.post_process_panoptic(
                        outputs, processed_sizes)[0]

            extracted_classes = []

            for info in result['segments_info']:
                if info["isthing"] == True:
                    cat_id = self.coco_classes['things_ids'][str(
                        info['category_id'])]
                    cat = self.coco_classes['things'][cat_id]
                    if cat is not None:
                        extracted_classes.append(cat)
                else:
                    stuff_id = self.coco_classes['stuffs_ids'][str(
                        info['category_id'])]
                    stuff = self.coco_classes['stuffs'][stuff_id]
                    if stuff is not None:
                        extracted_classes.append(stuff)
            counts = Counter(extracted_classes)
            return [(key, value) for key, value in counts.items()]
        except Exception as e:
            print(f"An error occurred on panoptic_predictor: {e}")
            return None

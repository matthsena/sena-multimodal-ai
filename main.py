import gc
import os
import sys
import time
import cv2
import database.sqlite as database
from utils.map_files import image_list as get_image_list
from itertools import combinations
from models.cv.vgg19 import extract_features, compare_features
from models.cv.ocr import OCRPredictor
from models.cv.inception3 import predict as inception3_predictor
from models.cv.resnet50 import predict as resnet50_predictor
from models.cv.panoptic import PanopticPredictor
import torch
from torch.cuda.amp import GradScaler, autocast

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

DISTANCE_THRESHOLD = 0.25
LANGS_TO_COMPARE = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

start_time = time.time()


def load_image(img_path: str):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(
                f"Erro ao carregar a imagem. Verifique se o caminho '{img_path}' está correto.")
            raise Exception("Imagem não encontrada.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        exit(1)


def image_distances(img1, img2):
    try:
        features_img1 = extract_features(img1)
        features_img2 = extract_features(img2)

        if features_img1 is None or features_img2 is None:
            print("Erro ao extrair características das imagens.")
            return None

        distance = compare_features(features_img1, features_img2)

        if distance is None:
            print("Erro ao comparar características das imagens.")
            return None

        return distance
    except Exception as e:
        print(f"Erro ao comparar características das imagens: {e}")
        return None


image_list = list(map(get_image_list, LANGS_TO_COMPARE))
image_list = list(filter(lambda x: x is not None, image_list))

db = database.SQLiteOperations()
ocr_predictor = OCRPredictor()
panoptic_predictor = PanopticPredictor()

scaler = GradScaler()
with autocast():
    for lang_path, images in image_list:
        for image in images:
            file_path = f'{lang_path}/{image}'
            img = load_image(file_path)

            features = extract_features(img)
            # check if alread exists
            features_list = features.tolist()
            feature_history = db.select_by_features(features_list)

            if feature_history is not None:
                db.upsert(file_path, features_list, feature_history['ocr'], feature_history[
                          'panoptic'], feature_history['inception_v3'], feature_history['resnet50'])
                print(
                    f"Imagem {file_path} já existe uma identica no banco de dados.")
            else:
                ocr_texts = ocr_predictor.predict(img)
                inception_classes = inception3_predictor(img)
                resnet_classes = resnet50_predictor(img)
                panoptic_classes = panoptic_predictor.predict(img)

                db.upsert(file_path, features_list, ocr_texts,
                          panoptic_classes, inception_classes, resnet_classes)

            gc.collect()
            torch.cuda.empty_cache()


list_to_compare = []
seen = set()

for (lang_path1, img_path1), (lang_path2, img_path2) in combinations(image_list, 2):
    for img1 in img_path1:
        for img2 in img_path2:
            sorted_items = tuple(sorted([lang_path1, lang_path2, img1, img2]))
            if sorted_items not in seen:
                seen.add(sorted_items)
                list_to_compare.append(
                    (f'{lang_path1}/{img1}', f'{lang_path2}/{img2}'))

end_time = time.time()
elapsed_time = (end_time - start_time) // 60
print(
    f"Total time taken: {elapsed_time} minutes and {(end_time - start_time) % 60:.2f} seconds")

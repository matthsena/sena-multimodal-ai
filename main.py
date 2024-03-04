import gc
import os
import time
import cv2
import pandas as pd
import database.sqlite as database
from utils.map_files import image_list as get_image_list
from itertools import combinations
from models.cv.vgg19 import FeatureExtractor
from models.cv.ocr import OCRPredictor
from models.cv.inception3 import predict as inception3_predictor
from models.cv.resnet50 import predict as resnet50_predictor
from models.cv.panoptic import PanopticPredictor
import torch
from torch.cuda.amp import GradScaler, autocast
from concurrent.futures import ThreadPoolExecutor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# TF_ENABLE_ONEDNN_OPTS=0

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

image_list = list(map(get_image_list, LANGS_TO_COMPARE))
image_list = list(filter(lambda x: x is not None, image_list))

db = database.SQLiteOperations()
ocr_predictor = OCRPredictor()
panoptic_predictor = PanopticPredictor()
vgg19 = FeatureExtractor()

scaler = GradScaler()
with autocast():
    for lang_path, folder_path, images in image_list:
        for image in images:
            file_path = f'{folder_path}/{image}'
            img = load_image(file_path)

            features = vgg19.extract_features(img)
            
            features_list = features.tolist()
            feature_history = db.select_by_features(features_list)

            if feature_history is not None:
                db.upsert(file_path, lang_path, features_list, feature_history['ocr'], feature_history[
                          'panoptic'], feature_history['inception_v3'], feature_history['resnet50'])
                print(
                    f"Imagem {file_path} já existe uma identica no banco de dados.")
            else:
                with ThreadPoolExecutor() as executor:
                    ocr_texts_future = executor.submit(ocr_predictor.predict, img)
                    inception_classes_future = executor.submit(inception3_predictor, img)
                    resnet_classes_future = executor.submit(resnet50_predictor, img)
                    panoptic_classes_future = executor.submit(panoptic_predictor.predict, img)
                
                ocr_texts = ocr_texts_future.result()
                inception_classes = inception_classes_future.result()
                resnet_classes = resnet_classes_future.result()
                panoptic_classes = panoptic_classes_future.result()

                db.upsert(file_path, lang_path, features_list, ocr_texts,
                          panoptic_classes, inception_classes, resnet_classes)

            gc.collect()
            torch.cuda.empty_cache()


list_to_compare = []
seen = set()

# for (lang_path1, img_path1), (lang_path2, img_path2) in combinations(image_list, 2):
#     for img1 in img_path1:
#         for img2 in img_path2:
#             sorted_items = tuple(sorted([lang_path1, lang_path2, img1, img2]))
#             if sorted_items not in seen:
#                 seen.add(sorted_items)
#                 list_to_compare.append(
#                     (f'{lang_path1}/{img1}', f'{lang_path2}/{img2}'))
for lang, path, photos in image_list:
    for other_lang, path, other_photos in image_list:
        if lang != other_lang:
            for photo in photos:
                for other_photo in other_photos:
                    sorted_items = tuple(
                        sorted([lang, other_lang, f'{path}/{photo}', f'{path}/{other_photo}']))
                    if sorted_items not in seen:
                        seen.add(sorted_items)
                        list_to_compare.append(
                            ((lang, other_lang), (f'{path}/{photo}', f'{path}/{other_photo}')))

print(list_to_compare)

result_list = []

with ThreadPoolExecutor() as executor:
    futures = []
    for (lang1, lang2), (img1, img2) in list_to_compare:
        f1 = db.select_feature_by_file_path(img1)
        f2 = db.select_feature_by_file_path(img2)
        future = executor.submit(vgg19.compare_features, f1, f2)
        futures.append((future, (lang1, lang2, img1, img2)))

    for future, (lang1, lang2, img1, img2) in futures:
        result_list.append({
            'article': 'copa',
            'original': lang1,
            'compare': lang2,
            'original_photo': img1,
            'compare_photo': img2,
            'distance': future.result()
        })

df_result = pd.DataFrame(result_list)
df_result.to_csv('result.csv', index=False)

end_time = time.time()
elapsed_time = (end_time - start_time) // 60
print(
    f"Total time taken: {elapsed_time} minutes and {(end_time - start_time) % 60:.2f} seconds")

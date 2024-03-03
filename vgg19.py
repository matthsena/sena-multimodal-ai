import cv2
from keras.applications.vgg19 import VGG19, preprocess_input
import keras.utils as image
import numpy as np
from scipy.spatial import distance

try:
  model = VGG19(weights='imagenet', include_top=False)
except Exception as e:
  print(f"[VGG19] Erro ao carregar o modelo: {e}")
  model = None

def compare_features(features_img1, features_img2):
    try:
        if features_img1 is None or features_img2 is None:
            print("[VGG19] Uma ou ambas as características são nulas.")
            return None
        return distance.cosine(features_img1, features_img2)
    except Exception as e:
        print(f"[VGG19] Erro ao comparar características: {e}")
        return None

def extract_features(img):
  try:
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    return features
  except Exception as e:
    print(f"[VGG19] Erro ao extrair características da imagem: {e}")
    return None
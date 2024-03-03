import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
model.eval()


def load_imagenet_classes():
    with open('./imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def decode_predictions(preds, top=3):
    classes = load_imagenet_classes()
    probabilities = F.softmax(preds, dim=1)[0]
    _, indices = torch.topk(probabilities, top)
    decoded_preds = [(classes[idx], probabilities[idx].item())
                     for idx in indices]
    return decoded_preds


def predict(image):
    # convert cv2 image to PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(x)

    return decode_predictions(preds, top=5)

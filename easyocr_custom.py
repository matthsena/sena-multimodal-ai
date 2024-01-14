import easyocr
import torch
import cv2

isGpuAvailable = torch.cuda.is_available()


def predict(image):
  try:
    reader = easyocr.Reader(
      ['en', 'pt', 'es', 'fr', 'de', 'it'], gpu=isGpuAvailable)
    result = reader.readtext(image)
    valuesAndProb = []

    for (bbox, text, prob) in result:
      (tl, _, br, _) = bbox
      tl = (int(tl[0]), int(tl[1]))
      br = (int(br[0]), int(br[1]))
      cv2.rectangle(image, tl, br, (0, 255, 0), 2)
      cv2.putText(image, text, (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
      valuesAndProb.append((text, prob))
    return image, valuesAndProb
  except Exception as e:
    print(f"An error occurred in OCR fn: {str(e)}")
    return None, None

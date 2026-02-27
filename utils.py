import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img, thresh

def extract_text_easyocr(image_or_processed):
    result = reader.readtext(image_or_processed)
    text = " ".join([res[1] for res in result])
    return text.strip()

"""
    Here EasyOCR extracts text from images and returns it as string
"""

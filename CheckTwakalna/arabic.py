import glob

import cv2
import pytesseract
from PIL import Image
import os

from imutils import contours
import pytesseract
import os

os.environ['TESSDATA_PREFIX'] = '.'

text = pytesseract.image_to_string('text-ara.jpg', lang='ara')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

language_path = 'C:\\Program Files\\Tesseract-OCR\\tessdata\\'
language_path_list = glob.glob(language_path+"*.traineddata")
text = pytesseract.image_to_string('2.png',lang='ara', config='--tessdata-dir')
print(text)
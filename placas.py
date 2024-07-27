import cv2
import numpy as np
import pytesseract
from PIL import Image

cap = cv2.VideoCapture('')
Ctexto = ''

while True:
    #
    ret, frame = cap.read()

    if ret == false:
        break

    cv2.rectangle(frame, (870, 750), (1070, 850))
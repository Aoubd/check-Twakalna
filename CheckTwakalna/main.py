import re
import time

import cv2
import pickle
import cvzone
import numpy as np
from ArabicOcr import arabicocr
import pytesseract
import easyocr

reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
lower_white = np.array([0, 0, 220])
upper_white = np.array([255, 255, 255])
lower = np.array([52, 0, 55])
upper = np.array([104, 255, 255])  # (These ranges will detect Yellow)
count = 1
timer = int(time.strftime('%S')) + 5

# Video feed
cap = cv2.VideoCapture('1.mp4')

with open('check', 'rb') as f:
    posList = pickle.load(f)

width, height = 400, 200


def check(imgPro, imgWhit):
    global count,timer
    out_image = 'out.jpg'
    for pos in posList:
        x, y = pos
        color = (0, 255, 0)
        thickness = 2
        # green
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        mask = cv2.inRange(imgPro, lower, upper)  # Masking the image to find our color

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask image
        res = cv2.bitwise_and(imgPro, imgPro, mask=mask)

        # Finding position of contour
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > 500:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    if w > 250 and (pos[0] + width) > (x + w - 20) and (pos[1] + height) > (y + h - 2) and h < 450:
                        # red drawing
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle
                        imgCrop = imgPro[y:y + height, x:x + width]
                        # cv2.imshow('imgCrop', img)
                        cv2.imwrite('out1.jpg', img[y:y + h, x:x + w])
                        #print(timer,'\n',time.strftime('%S'))

                        # read img
                        if int(time.strftime('%S')) == timer:
                            timer = int(time.strftime('%S')) + 5
                            text = reader.readtext('out1.jpg')
                            for ele in text:
                                if ele[1] == 'محضن':
                                    print('محصن ')
                                    break
                                elif ele[1] == 'مدئن':
                                    print('محصن ')
                                    break
                                elif ele[1] == 'محقن':
                                    print('محصن ')
                                    break
                                elif ele[1] == 'محضن':
                                    print('محصن ')
                                elif ele[1] == 'محصن':
                                    print('محصن ')
                                elif ele[1] == 'محقئن':
                                    print('محصن ')
                                    break
                                elif ele[1] == 'مخضن':
                                    print('محصن ')
                                    break
                                else:
                                    print('error')
                                    break

        # find white color

        maskx = cv2.inRange(imgWhit, lower_white, upper_white)  # Masking the image to find our color
        mask_contours, xhierarchy = cv2.findContours(maskx, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask imge
        resx = cv2.bitwise_and(imgWhit, imgWhit, mask=maskx)
        cv2.drawContours(img, mask_contours, 0, (0, 255, 0), 3)
        # Finding position of all contours
        if len(mask_contours) != 0:
            for xmask_contour in mask_contours:
                if cv2.contourArea(xmask_contour) > 500:
                    x, y, w, h = cv2.boundingRect(xmask_contour)
                    if w > 250 and (pos[0] + width) > (x + w) and (pos[1] + height) > (y + h) and h < 450:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle
                        cv2.imwrite('out1.jpg', img[y:y + h, x:x + w])

                        if int(time.strftime('%S')) == timer:
                            timer = int(time.strftime('%S')) + 5
                            text = reader.readtext('out1.jpg')
                            print(text)
                            for ele in text:
                                if ele[1] == 'Incomplete VaceinuUon':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'Incomplate Vaccinaton':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'Incomplete Vacenution':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'Incomplete Vacenatlon':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'Incompletc Viccnunon':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'Incomplcto Veconauon':
                                    print('Incomplate Vaccinaton')
                                    break
                                elif ele[1] == 'IncomplctoViccnton':
                                    print('Incomplate Vaccinaton')
                                    break
                                else:
                                    print("Please Try again")
                                    break
                        # read img

        # cv2.imshow("mask image", mask)  # Displaying mask image


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 1)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    imgWhite = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    check(imgColor, imgWhite)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgGray)
    cv2.waitKey(10)

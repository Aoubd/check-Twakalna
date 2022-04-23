import cv2
import pickle
import cvzone
import numpy as np
from ArabicOcr import arabicocr
from pyzbar.pyzbar import decode


lower = np.array([52, 0, 55])
upper = np.array([104, 255, 255])  # (These ranges will detect Yellow)
count = 1
# Video feed
cap = cv2.VideoCapture('1.mp4')

with open('check', 'rb') as f:
    posList = pickle.load(f)

width, height = 480, 170


def check(imgPro):
    global count
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
                    if w > 250 and (pos[0] + width) > (x + w) and (pos[1] + height) > (y + h):
                        # read barcode
                        for barcode in decode(img):
                            myData = barcode.data.decode('utf-8')
                            print(myData)
                            pts = np.array([barcode.polygon], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(img, [pts], True, (255, 0, 255), 5)
                            pts2 = barcode.rect
                            cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255),
                                        2)
                        # end barcode

                        # red drawing
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle
                        imgCrop = imgPro[y:y + height, x:x + width]
                        cv2.imshow('imgCrop', img)
                        cv2.imwrite('out1.jpg',img[y:y + height, x:x + width] )
                        # read img
                        if count == 1:
                            count = 2
                            results = arabicocr.arabic_ocr(str('out1.jpg'), out_image)
                            # check  محصن او غير محصن
                            if str(results[0][1]) == 'محقن':
                                print('محصن')

                            elif str(results[0][1]) != 'محقن':
                                print('غير محصن')
                            else:
                                print('Please try again ')


        cv2.imshow("mask image", mask)  # Displaying mask image
        cv2.imshow('res', res)


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    check(imgColor)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgMedian)
    cv2.waitKey(10)

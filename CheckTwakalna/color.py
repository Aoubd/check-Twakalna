import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([52, 0, 55])
upper = np.array([104, 255, 255])  # (These ranges will detect Yellow)

# Capturing webcam footage
webcam_video = cv2.VideoCapture('1.mp4')

while True:
    success, video = webcam_video.read()  # Reading webcam footage

    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    mask = cv2.inRange(img, lower, upper)  # Masking the image to find our color
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask image
    res = cv2.bitwise_and(video, video, mask=mask)
    cv2.drawContours(img, mask_contours, 0, (0, 255, 0), 3)
    #Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle


    cv2.imshow("mask image", mask)  # Displaying mask image

    cv2.imshow("window", video)  # Displaying webcam image
    cv2.imshow('res', res)
    cv2.waitKey(1)

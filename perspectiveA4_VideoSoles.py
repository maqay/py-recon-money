import cv2
import numpy as np
#import imutils
import datetime
from global_utils import *


cap = cv2.VideoCapture(0)
dt_1 = datetime.datetime.now()

while True:

    ret, frame = cap.read()
    #print(frame.shape)

    if ret == False: break
    # frame = imutils.resize(frame, width=720)
    cv2.imshow('frame', frame)

    imagen_A4 = roi(frame, ancho=720, alto=509)

    if imagen_A4 is not None:
        gray = cv2.cvtColor(imagen_A4, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        #contornos
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cant_circles = 0
        for cnt in contours:
            nro_edges = cv2.approxPolyDP(cnt, .01 * cv2.arcLength(cnt, True), True)

            k=cv2.isContourConvex(nro_edges)
            area = cv2.contourArea(cnt)

            if k and area>500:
                cv2.drawContours(imagen_A4, [cnt],0, (0, 255, 0), 2)
                cant_circles = cant_circles + 1
                '''
                print('circle:', cant_circles)
                print('edges:->', len(nro_edges))
                print('area:->', area)
                '''

#        print('cant_circles', cant_circles)



        cv2.imshow('bin_img', bin_img)
        cv2.imshow('imagen_A4', imagen_A4)
        #cv2.waitKey(0)



        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

dt_2 = datetime.datetime.now()
demora = dt_2 - dt_1
print('demora:', demora)
cap.release()
cv2.destroyAllWindows()

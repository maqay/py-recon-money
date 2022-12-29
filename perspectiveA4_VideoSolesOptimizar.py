import cv2
import numpy as np
import datetime
from global_utils import *

cap = cv2.VideoCapture(1)
dt_1 = datetime.datetime.now()

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

    if ret == False:
        print("False")
        break

    cv2.imshow('frame', frame)
    cv2.imshow('bin_img', bin_img)

    #imagen_A4 = roi(frame, ancho=720, alto=509)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


dt_2 = datetime.datetime.now()
demora = dt_2 - dt_1
print('demora:', demora)
cap.release()
cv2.destroyAllWindows()

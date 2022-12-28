#https://dev.to/tinazhouhui/coin-detection-discovering-opencv-with-python-1ka1
import cv2
import numpy as np

def hough_circle_detection(coins, min_r, max_r):
    # turn original image to grayscale
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    # blur grayscale image
    blurred = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(
        blurred,  # source image (blurred and grayscaled)
        cv2.HOUGH_GRADIENT,  # type of detection
        1,  # inverse ratio of accumulator res. to image res.
        40,  # minimum distance between the centers of circles
        param1=50,  # Gradient value passed to edge detection
        param2=30,  # accumulator threshold for the circle centers
        minRadius=min_r*2,  # min circle radius
        maxRadius=max_r*2,  # max circle radius
    )

def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def roi(image, ancho, alto):
    imagen_alineada = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow('th', th)

    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) == 4:
            puntos = ordenar_puntos(approx)
            pts1 = np.float32(puntos)
            pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_alineada = cv2.warpPerspective(image, M, (ancho, alto))
    return imagen_alineada
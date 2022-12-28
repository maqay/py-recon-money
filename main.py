import cv2
import numpy as np
import matplotlib.pyplot as plt

bills = cv2.imread("/home/michael/images/billete1.jpeg",0)
print(bills.shape)


ret,thresh = cv2.threshold(bills,80,255,cv2.THRESH_BINARY)

contours, hierarchy=cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(bills.shape)

print('contours:', len(contours))

cv2.drawContours(external_contours,contours,-1,(0,0,255),2)

#for i in range(len(contours)):
#        cv2.drawContours(external_contours,contours,i,(0,255,0),-1)

plt.imshow(thresh,cmap='gray')

plt.show()
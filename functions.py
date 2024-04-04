import numpy as np
import cv2


def biggestContour(contours):
    biggest=np.array([])
    max_area=0

    for i in contours:
        area=cv2.contourArea(i)
        if area>5000:
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(approx)==4:
                biggest=approx
                max_area=area

    return biggest,max_area

def reorder(points):
    points=points.reshape((4,2))
    new_points=np.zeros((4,1,2),dtype=np.int32)
    add=points.sum(1)

    new_points[0]=points[np.argmin(add)]
    new_points[3]=points[np.argmax(add)]

    diff=np.diff(points,axis=1)
    new_points[1]=points[np.argmin(diff)]
    new_points[2]=points[np.argmax(diff)]

    return new_points

def drawRectangle(img, biggest, thickness):
    cv2.line(img,(biggest[0][0][0], biggest[0][0][1]),(biggest[1][0][0],biggest[1][0][1]),thickness)
    cv2.line(img,(biggest[0][0][0], biggest[0][0][1]),(biggest[2][0][0],biggest[2][0][1]),thickness)
    cv2.line(img,(biggest[3][0][0], biggest[3][0][1]),(biggest[2][0][0],biggest[2][0][1]),thickness)
    cv2.line(img,(biggest[3][0][0], biggest[3][0][1]),(biggest[1][0][0],biggest[1][0][1]),thickness)
import cv2
import numpy as np
import funtions as func

def on_trackbar1(value):
    pass

def on_trackbar2(value):
    pass  

cv2.namedWindow('Trackbars')
cv2.createTrackbar('Lower', 'Trackbars', 0, 255, on_trackbar1)
cv2.createTrackbar('Higher', 'Trackbars', 200, 400, on_trackbar2)



while True:
    image_path = "Images/Dollar3.jpg"
    img =cv2.imread(image_path)
    img = cv2.resize(img, (500, 330))

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)

    trackbar_value1 = cv2.getTrackbarPos('Lower', 'Trackbars')
    trackbar_value2 = cv2.getTrackbarPos('Higher', 'Trackbars')
  
    imgThreshold=cv2.Canny(imgBlur,trackbar_value1,trackbar_value2)

    kernel=np.ones((5,5))
    imgDial=cv2.dilate(imgThreshold,kernel,iterations=2)
    imgThreshold=cv2.erode(imgDial,kernel,iterations=1)
    

    imgContour=img.copy()
    imgBigcontour=img.copy()

    contours, heiarchy=cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgBigcontour,contours,-1,(0,255,0),2)

    biggest,maxArea=func.biggestContour(contours)

    

    if biggest.size!=0:
        biggest=func.reorder(biggest)
        cv2.drawContours(imgContour,biggest,-1,(0,255,0),20)
        imgBigcontour=func.drawRectangle(imgBigcontour,biggest,2)
        pst1=np.float32(biggest)
        pst2=np.float32([[0,0], [img.shape[1],0],[0,img.shape[0]],[img.shape[1],img.shape[0]]])
        matrix=cv2.getPerspectiveTransform(pst1,pst2)
        imgWrapColored=cv2.warpPerspective(img,matrix,(img.shape[1],img.shape[0]))



    cv2.imshow("Image", imgWrapColored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




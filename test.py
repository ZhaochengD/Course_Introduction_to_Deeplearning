import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.medianBlur(gray, 11)

        scale_percent = 1
        width = int(gray2.shape[1] * scale_percent )
        height = int(gray2.shape[0] * scale_percent )
        resized_gray = cv2.resize(gray2, (width, height), interpolation = cv2.INTER_AREA)
        # write the flipped frame
        # out.write(frame)

        kernel = np.ones((3,3),np.uint8)
        morph_gray = cv2.morphologyEx(resized_gray, cv2.MORPH_GRADIENT, kernel)

        res,Otsu_gray = cv2.threshold(morph_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cnts = cv2.findContours(Otsu_gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        screenCnt = False
        for c in cnts:
        ### Approximating the contour
        #Calculates a contour perimeter or a curve length
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            area = cv2.contourArea(c)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
            if len(approx) == 4 and area > 0.1 * frame.shape[0] * frame.shape[1]:
                screenCnt = approx
                break
            
        if screenCnt is not False:
            cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
            warped = four_point_transform(frame, screenCnt.reshape(4, 2))
            cv2.imshow('warped', warped)
        image_e = cv2.resize(frame,(frame.shape[1],frame.shape[0]))
        cv2.imshow('img', image_e)
        # cv2.imwrite('image_edge.jpg',image_e)
    #except:
        #pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# import utils
# import cv2
# import hypertools as hyp
#
# img = cv2.imread('./Images/flag.jpeg')
# img = cv2.resize(img, (300,300))
# # ret, img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
#
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# cv2.imshow("img", img)
# cv2.waitKey()
#
# pts = utils.image_to_points(img)
# pts_hsv = utils.image_to_points(img_hsv)
#
# hyp.plot(pts, '.', group=pts_hsv[:,0], save_path='./t.png')

import numpy as np
import cv2

cap = cv2.VideoCapture('http://172.30.22.40:8080/')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        print('Error frame')
        continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
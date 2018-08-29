import utils
import cv2
import hypertools as hyp

img = cv2.imread('./Images/flag.jpeg')
img = cv2.resize(img, (300,300))
# ret, img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("img", img)
cv2.waitKey()

pts = utils.image_to_points(img)
pts_hsv = utils.image_to_points(img_hsv)

hyp.plot(pts, '.', group=pts_hsv[:,0], save_path='./t.png')
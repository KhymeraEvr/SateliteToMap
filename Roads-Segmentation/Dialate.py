import cv2
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

img=cv2.imread('DilateImput/output2.png')
img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
cv2.imwrite('DilateImput/out/outputBW.png', eroded)
selem = disk(3)
eroded = erosion(img_bw, selem)
cv2.imwrite('DilateImput/out/outputBW.png', eroded)
img_bw = eroded

se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

mask = np.dstack([mask, mask, mask]) / 255
out = img * mask

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('DilateImput/out/output.png', out)

#kernel = np.ones((12,12), np.uint8)  # note this is a horizontal kernel
#d_im = cv2.dilate(img, kernel, iterations=1)
#e_im = cv2.erode(d_im, kernel, iterations=1)
#cv2.imwrite('dialate.jpg', e_im)
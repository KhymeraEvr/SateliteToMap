import numpy as np
import cv2

#inputFolder = 'SegmentRemoval/Input/'
inputFolder = 'Predictions/output/'
ClosingFolder = 'SegmentRemoval/Closing/'
Closing2Folder = 'SegmentRemoval/Closing2/'
DilateFolder = 'SegmentRemoval/Dilate/'
RemoveSegment = 'SegmentRemoval/RemovedSegment/'
TreshGray = 'SegmentRemoval/TreshGray/'
Final = 'SegmentRemoval/Final/'

def is_contour_bad(c): # Decide what I want to find and its features
    peri = cv2.contourArea(c) # Find areas
    return peri < 100


### DATA PROCESSING ###
def RemoveSegments(fileName):
    image = cv2.imread( inputFolder + fileName) # Load a picture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    # Convert to binary image (all values above 50 are converted to 1 and below to 0)
    ret, thresh_gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  #Tresh

    cv2.imwrite(TreshGray + fileName, thresh_gray) # Plot

    # Use "close" morphological operation to close the gaps between contours
    # https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))); #close1

    cv2.imwrite(ClosingFolder + fileName, thresh_gray) # Plot


    #Find contours on thresh_gray, use cv2.RETR_EXTERNAL to get external perimeter
    (cnts, _) = cv2.findContours(thresh_gray, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE) # Find contours: a curve joining all the continuous points (along the boundary), having same color or intensity

    image_cleaned = (image*0).astype('uint8')


    # Loop over the detected contours
    for c in cnts:
        # If the contour satisfies "is_contour_bad", draw it on the mask
        if not is_contour_bad(c):
            # Draw black contour on gray image, instead of using a mask
            print('drawing')
            cv2.drawContours(image_cleaned, [c], -1, (255,255,255), 1)


    #cv2.imshow("Adopted mask", mask) # Plot
    #cv2.imshow("Cleaned image", image_cleaned) # Plot
    cv2.imwrite(RemoveSegment + fileName, image_cleaned, [cv2.IMREAD_ANYDEPTH ]) # Write in a file

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(image_cleaned, kernel, iterations=1) #dilate
    cv2.imwrite(DilateFolder + fileName, dilation)
    img_bw = dilation

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1) # close
    cv2.imwrite(Closing2Folder + fileName, mask)
    return fileName

    #mask = (np.dstack([mask, mask, mask]) / 255).astype(np.uint8)
    #out = image_cleaned.astype(np.uint8) * mask

    #3cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite(Final + fileName, out)

#RemoveSegments('19_pred.png')
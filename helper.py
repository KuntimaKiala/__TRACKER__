
import cv2 as cv
import numpy as np
from constants import lk_params, feature_params

# contain all the helper functions, common-generics functions to all the project



def undistort(image, cameraMatrix,distCoeffs, scale = 0) :
    """
    Input :
    image : the image to be processed
    cameraMatrix :  3x3 ndarray 
    distCoeffs   : 1x5 ndarray conatining the distortion coefficients, they can be 4, 5, 8,12, or 14. (in our case )
    undistortedImage : the image after processing, without the lens distortions
    """
        
    
    #getting the images size
    h, w = image.shape[:2]
    # from the intrinsic get the optimum 
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
    # perform the undistortion tranformation on the image
    undistortedImage = cv.undistort(image,cameraMatrix, distCoeffs, scale, newCameraMatrix )
    x,y,w,h = roi #
    undistortedImage  = undistortedImage[y:y+h, x:x+w]
    undistortedImage = cv.resize(undistortedImage,(1280,720))

    return undistortedImage 


def feature_extraction_KLT(img0, img1, p0, lk_params) : 
    p1, _st, _err =cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

    return p1, _st, _err



def featureExtraction(grayimage, mask = []) :

    orb = cv.ORB_create()  #cv2.SIFT_create() #


    # selection of good features extraction (corner points), KLT approach 500
    
    if mask == [] :
        pts = cv.goodFeaturesToTrack(grayimage,**feature_params)
    else :
        pts = cv.goodFeaturesToTrack(grayimage, mask = mask,**feature_params )
    
    # keypoint creation
    kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]

    # keypoint and descriptor extraction from corner points from KTL approach
    
    kps, des = orb.compute(grayimage, kps) 
    p = np.array([[(kp.pt[0], kp.pt[1])] for kp in kps], dtype=np.float32) 

    return  kps, des, p



def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

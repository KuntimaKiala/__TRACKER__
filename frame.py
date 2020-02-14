
import cv2
from helper import feature_extraction_KLT, featureExtraction, normalize
from constants import lk_params, nn_match_ratio, MIN_MATCH_COUNT, inlier_threshold
import numpy as np
from math import sqrt
from constants import cameraMatrix

def match_frames(
                 f1, 
                 f2,
                 nn_match_ratio=nn_match_ratio, 
                 MIN_MATCH_COUNT = MIN_MATCH_COUNT, 
                 inlier_threshold=inlier_threshold ) :
    


    # brute force matcher 
    Matcher =  cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) #cv2.BFMatcher(cv2.NORM_L1, crossCheck=False) 
    # matches 
    matches = Matcher.knnMatch(f1.des,f2.des,k=2)

    # Lowe's ratio test

    
    good = []
    for m, n in matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append(m) 

    
    if len(good) > MIN_MATCH_COUNT:
        
       
        dst_pts = np.float32([f2.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.float32([f1.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        
         
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)
        if np.any(homography)== None :
            return None, None, None,None
        
    
    else:
        print(
            "Not enouth match are found -%d/%d" % (len(good), MIN_MATCH_COUNT))
        
        return None, None, None,None


    # [homography check]
    inliers1 = []
    inliers2 = []
    idx = {0:[], 1:[]}
    good_matches = []


    # Distance threshold to identify inliers with homography check
    
   
    for i, m in enumerate(good):
        col = np.ones((3,1), dtype=np.float64)
        col[0:2,0] = f1.kp[m.queryIdx].pt

       
        col = np.dot(homography, col) 
        
        col /= col[2,0]
       
        dist = sqrt(
            pow(col[0,0] - f2.kp[m.trainIdx].pt[0],2) + pow(col[1,0] - f2.kp[m.trainIdx].pt[1],2))

        if dist < inlier_threshold:
            
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(f1.kp[good[i].queryIdx])
            idx[0].append(good[i].queryIdx)

            inliers2.append(f2.kp[good[i].trainIdx]) 

                
            idx[1].append(good[i].trainIdx)



    
    
   
    
    return inliers1, idx[0], inliers2, idx[1]


class Frame() :

    def __init__(self, img0,img1, mapp,px, mask, tid=None) : 
        self.start_rating = False 
        self.K = cameraMatrix
        if img0 is not None : 
            
            gray_image0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY) 
            gray_image1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            self.p, _,_ = feature_extraction_KLT(gray_image0,gray_image1,px,lk_params)
            self.kp, self.des, self.pts = featureExtraction(gray_image0, mask )
            
        else :
            self.kp, self.des, self.pts = None, None, None 
        
        self.id = tid if tid is not None else mapp.add_frame(self)
       
           
    

    
    
    def annotate(self, img) :

        pass


    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            self._Kinv = np.linalg.inv(self.K)
        return self._Kinv

    # normalized keypoints
    @property
    def kps(self):
        
        if not hasattr(self, '_kps'):
            self._kps = normalize(self.Kinv, self.kp)
        return self._kps

import cv2 as cv
import numpy as np
from constants import lk_params, feature_params
import csv

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
    

    return  np.array([(kp.pt[0], kp.pt[1]) for kp in kps]) , des, kps



def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)



class Rating() :

  def __init__(self) :
      self.nbKeypoints = [0]*2
      self.start_rating = False

  def rate_of_point_matching(self,nbKeypoints) :

      if self.start_rating == False : 
          self.nbKeypoints[0] = nbKeypoints
          self.nbKeypoints[1] = nbKeypoints
          self.start_rating = True

      else :
          self.nbKeypoints[1] = nbKeypoints


      self.rate = self.nbKeypoints[1]*100.0/self.nbKeypoints[0]

      return self.rate




def writting_file(filename, vector_points):
    
    with open(filename, mode='a') as file:

        csv_writer = csv.writer(file, delimiter=',')

        csv_writer.writerow(vector_points) 



import os

def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


def fundamentalToRt(F):
    
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
    if t[2] < 0:
        t *= -1
  
    # TODO: UGLY!
    if os.getenv("REVERSE") is not None:
        t *= -1
        print('R', R)
    return np.linalg.inv(poseRt(R, t))


def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

# from https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
class EssentialMatrixTransform(object):
  def __init__(self):
    self.params = np.eye(3)

  def __call__(self, coords):
    coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
    return coords_homogeneous @ self.params.T

  def estimate(self, src, dst):
    #print(src.shape)
    assert src.shape == dst.shape
    assert src.shape[0] >= 8
    
    # Setup homogeneous linear equation as dst' * F * src = 0.
    A = np.ones((src.shape[0], 9))
    A[:, :2] = src
    A[:, :3] *= dst[:, 0, np.newaxis]
    A[:, 3:5] = src
    A[:, 3:6] *= dst[:, 1, np.newaxis]
    A[:, 6:8] = src

    # Solve for the nullspace of the constraint matrix.
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    # Enforcing the internal constraint that two singular values must be
    # non-zero and one must be zero.
    U, S, V = np.linalg.svd(F)
    S[0] = S[1] = (S[0] + S[1]) / 2.0
    S[2] = 0
    self.params = U @ np.diag(S) @ V
    
    return True
    
  def residuals(self, src, dst):
    # Compute the Sampson distance.
    src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
    dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

  
    F_src = self.params @ src_homogeneous.T
    Ft_dst = self.params.T @ dst_homogeneous.T
    
    dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)
  
    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                       + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)
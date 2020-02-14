
#from frame import match_frames
import numpy as np
from helper import Rating, poseRt, fundamentalToRt, EssentialMatrixTransform
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS 
import cv2
from skimage.measure import ransac
def match_frame(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]

            # be within orb distance 32
            if m.distance < 32:
                # keep around indices
                # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    # no duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    #print('ret',len(ret))
    assert len(ret) >= 8
    
    #print(len(ret))
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=RANSAC_RESIDUAL_THRES,
                            max_trials=RANSAC_MAX_TRIALS)


    
    
 
    print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
    return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

class Map() :

    def __init__(self) :

        self.max_frame = 0
        self.frames = []
        self.key_frames =  []
        self.r = Rating()
        self.rate = 0.0
        self.counter = 0
        self.interval_remove_kf = 0




    def add_frame(self, frame) :
        ret = self.max_frame
        self.frames.append(frame)
        self.max_frame += 1
        return ret


    def add_key_frame(self) :
        


        
        f1 = self.frames[-2]
        f2 = self.frames[-1] 
        

        
        #print(self.counter, self.interval_remove_kf + 5)
        if self.counter > self.interval_remove_kf + 5 :
            #self.key_frames = [ x for i, x in enumerate(self.key_frames[:-1]) if i%2 != 0 ]

            self.interval_remove_kf = len(self.key_frames)
            self.counter = len(self.key_frames)
            #print('kF =', len(self.key_frames))
        if self.rate < 20 :
            self.r.start_rating = False
            self.key_frames.append(f1) 
            self.counter += 1

        good, _, _ = match_frame(self.key_frames[-1], f2) 
        if np.all(good) == None :
            good = []
        self.rate = self.r.rate_of_point_matching(len(good))
        
        print('rate =', self.rate)
        


        

import cv2
from helper import feature_extraction_KLT, featureExtraction
from constants import lk_params
class Frame() :

    def __init__(self, img0,img1, mapp,px, mask, tid=None) : 
        self.start_rating = False
        if img0 is not None : 
            
            gray_image0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY) 
            gray_image1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            self.p, _,_ = feature_extraction_KLT(gray_image0,gray_image1,px,lk_params)
            self.kp, self.des, self.pts = featureExtraction(gray_image0, mask )
            print('kp =', self.p.shape)
            print('p0 =', self.pts.shape)
           
            

        else :
            self.kp, self.des, self.pts = None, None, None 
        
        self.id = tid if tid is not None else mapp.add_frame(self)
        mapp.add_key_frame()
           
    

    
    
    def annotate(self, img) :

        pass
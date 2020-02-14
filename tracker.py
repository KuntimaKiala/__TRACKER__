


import cv2 

import numpy as np
from copy import copy 
from time import sleep
from shutil import rmtree as remove
from constants import *
from helper import undistort, featureExtraction, feature_extraction_KLT, draw_str, writting_file 
from pointmap import Map, match_frame
from frame import Frame, match_frames




class Tracker() :

    def __init__(self, video_camera):
        self.track_len = 10
        self.detect_interval = 15
        self.tracks = []
        self.cam = video_camera
        self.mapp = Map() 
        self._init_ = 5999
        self._frame_id = 0
        self.prev_gray = None
        self._initial_frame = None
        self.counter = 0
        self.color_id = 0
        self.c = 0


    def __process_frame__(self) :
        

        while True :

            # select which part of the video to start
            if self._frame_id < self._init_ :
                self._frame_id = self._init_
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, self._frame_id) 
              

            # read the frames 
            _ret, frame = self.cam.read()

            # remove distortions 
            #frame = undistort(frame,cameraMatrix, distCoeffs)
            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
           
            if len(self.tracks) > 0 :

                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2) 

                frame_obj = Frame(self._initial_frame, frame, self.mapp, p0, mask = []) 
                
                if frame_obj.id == 0 :
                    self._initial_frame  = frame
              
                    continue 

                
                img0, img1 = self.prev_gray, frame_gray 

                 
                #f1 = self.mapp.frames[-2]
                #f2 = self.mapp.frames[-1] 

                #f1 = self.mapp.frames[-1]
                #f2 = self.mapp.frames[-2]

                #match_frame(f1,f2)
                self.mapp.add_key_frame()
            
                
                # p1 : points matching from previous image and the new one, using last points p0
                p1, _st, _err = feature_extraction_KLT(img0, img1, p0, lk_params)
               
                # p0r : points matching from new image and the previous image, using new points p1 
                p0r, _st, _err = feature_extraction_KLT(img1, img0, p1, lk_params)
             
                # check the axis distance between those points (Not actulally a distance )
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                
                # requirement to keep following a track
 
                good = (d < 0.00899) & (d>0)

                new_tracks = []

                for tr, (x, y), good_flag in copy(zip(self.tracks, p1.reshape(-1, 2), good)):
                    if not good_flag:
                        continue
     
                    tr.append((x, y)) 
                    
                
                    if len(tr) > self.track_len:
                        del tr[0]
                        
            
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)

                
                self.tracks = new_tracks 

                
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, color[self.color_id]) #(0, 255, 0)
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                

            if self._frame_id % self.detect_interval == 0:

                # mask is the region of interest where the feature should be detected
                mask = np.zeros_like(frame_gray)

                mask[:] = 255

                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)

                p,_, _ = featureExtraction(frame_gray, mask)
                
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):

                        self.tracks.append([(x, y)]) 
                    
                self.color_id += 1

            # key board stroke
            ks = cv2.waitKey(1)
            if ks & 0xFF == ord('q') :

                break

            if ks == 27 :
                break

            cv2.imshow('Display', vis)

            self._frame_id += 1
            self.prev_gray = frame_gray
            self._initial_frame = frame
            

def main():
    import sys
    
    try :
        
    
        video_src = sys.argv[1]

        cap = cv2.VideoCapture(video_src)

        if cap is None or not cap.isOpened() :
            print('Warning: unable to open video source: ', video_src)
        else :
            print('Warning: successfully video source open : ', video_src)
            cam = cap

    except:
        video_src = video
        
        cap = cv2.VideoCapture(video_src)

        if cap is None or not cap.isOpened() :
            print('Warning: unable to open video source: ', video_src)
        else :
            print('Warning: successfully video source open : ', video_src)
            cam = cap
        

    
    Tracker(cam).__process_frame__()
    remove("./__pycache__")
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

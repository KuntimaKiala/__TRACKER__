
from frame import match_frames
import numpy as np
from helper import Rating
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
        
        print(self.counter, self.interval_remove_kf + 5)
        if self.counter > self.interval_remove_kf + 5 :
            self.key_frames = [ x for i, x in enumerate(self.key_frames[:-1]) if i%2 != 0 ]

            self.interval_remove_kf = len(self.key_frames)
            self.counter = len(self.key_frames)
            print('kF =', len(self.key_frames))
        if self.rate < 20 :
            self.r.start_rating = False
            self.key_frames.append(f1) 
            self.counter += 1

        good, _, _, _ = match_frames(self.key_frames[-1], f2) 
        self.rate = self.r.rate_of_point_matching(len(good))
        
        #print('rate =', self.rate)
        


        
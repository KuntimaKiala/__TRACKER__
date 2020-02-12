

class Map() :

    def __init__(self) :

        self.max_frame = 0
        self.frames = []
        self.key_frames =  []
        self.key_frames_id = 0 



    def add_frame(self, frame) :
        ret = self.max_frame
        self.frames.append(frame)
        self.max_frame += 1
        return ret


    def add_key_frame(self) :
        
        # match first frame descriptor with the rest
        # if you get less than  20% of match then remove all the point 
        # in between add 1 to  key_frames_id

        pass




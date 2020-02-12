import cv2 as cv
import numpy as np 

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 1,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.1,
                       minDistance = 15 )


distCoeffs = np.array([ -0.38846448, 0.12976164, 0.0, 0.0, 0.0 ])

cameraMatrix = np.array([[ 903.69745, 0.0, 634.7149],
                        [0.0, 914.1482,  384.99857 ],
                        [0.0,     0.0,    1.0      ]])


color = [(255,0,0), 
         (0,255,0),
         (0,0,255),
         ( 0,0,0),
         (255,255,255),
         (255,255,0),
         (0,255,255),
         (255,0,255)
         ]*100


video ="/media/ExtHDD1.8T/DataSets/moove_acquisition_circuit_versaille_vedecom/DonneesVersailles/_AZIZ/DATA/1_000BABB442A8_20190502_094452/RecFile_1_20190502_094452_bypass_video_7_output_1.avi"

nn_match_ratio = 0.4 
MIN_MATCH_COUNT = 0
inlier_threshold= 10
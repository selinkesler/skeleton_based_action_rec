#!/usr/bin/env python3
# Live Test ehpi ofp prod
import os
import glob

import cv2
import matplotlib
matplotlib.use('Agg') # due to gtk version problems as a solition
import numpy as np

if __name__ == '__main__':
    

##########################################################################################################################
#                                                     1. DATASET 
##########################################################################################################################
    
    s = 0    
    a = 0
    action_ID = 0
    for class_name in glob.glob('/home/xavier1/catkin_ws/src/gui/src/*'):
        print('CLASS NAME : ', class_name)
        str_class_name = str(class_name)
        for name in glob.glob(class_name + '/*'):
            print('NAME : ',name)

            if (name.find('utils') != -1):
                print('CALISIIIII')
    
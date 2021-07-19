#!/usr/bin/env python3
from collections import deque
from operator import itemgetter
from typing import Dict, List, Tuple
import os

from ehpi_action_recognition.config import pose_resnet_config, pose_visualization_config, ehpi_model_state_file_5, ehpi_model_custom_4_SGD, ehpi_model_custom_4_ADAM

import cv2
import matplotlib
matplotlib.use('Agg') # due to gtk version problems as a solition
import numpy as np
import time
import torch.utils.data.distributed
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action_custom_class import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.image_content_buffer import ImageContentBuffer
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import get_joints_jhmdb
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import FeatureVecProducerEhpi
from nobos_commons.input_providers.camera.webcam_provider import WebcamProvider
from nobos_commons.input_providers.camera.webcam_provider_USBCAM import WebcamProvider_USBCAM
from nobos_commons.input_providers.camera.img_dir_provider import VideoDirProvider
from nobos_commons.input_providers.camera.img_dir_provider import VideoDirProviderAsync
from nobos_commons.input_providers.camera.img_dir_provider import ImgDirProvider
from nobos_commons.tools.fps_tracker import FPSTracker
from nobos_commons.tools.log_handler import logger
from nobos_commons.tools.pose_tracker import PoseTracker
from nobos_commons.visualization.detection_visualizer import draw_bb, draw_bb_stalker
from nobos_commons.visualization.pose2d_visualizer import get_human_pose_image
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from nobos_torch_lib.models.pose_estimation_2d_models import pose_resnet
from scipy.special import softmax

from ehpi_action_recognition.configurator import setup_application
from ehpi_action_recognition.networks.action_recognition_nets.action_rec_net_ehpi import ActionRecNetEhpi
from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet ## for YOLOV5 pose2d_net_resnet_YOLOv5 instead of pose2d_net_resnet

from nobos_commons.data_structures.human import Human
from nobos_commons.tools.skeleton_converters.skeleton_converter_coco_to_stickman import SkeletonConverterCocoToStickman

import json
import glob

if __name__ == '__main__':
    setup_application()

    skeleton_type = SkeletonStickman
    image_size = ImageSize(width=1280, height=720)
    camera_number = 0
    fps = 30
    
    # for saving video 
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('/Users/sk82620/Pictures/Camera Roll/chair.mp4', fourcc, 20.0, (1280, 720))
    
    # Pose Network
    # pose_model = pose_resnet.get_pose_net(pose_resnet_config)
    # logger.info('=> loading model from {}'.format(pose_resnet_config.model_state_file))
    # pose_model.load_state_dict(torch.load(pose_resnet_config.model_state_file))
    # pose_model = pose_model.cuda()
    # pose_model.eval()

    with open('./networks/pose_estimation_2d_nets/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    pose_net = Pose2DNetResnet(skeleton_type,human_pose)
    pose_tracker = PoseTracker(image_size=image_size, skeleton_type=skeleton_type)  

    feature_vec_producer = FeatureVecProducerEhpi(image_size, get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
    
    
    s = 0    
    a = 0
    for class_name in glob.glob('/home/xavier1/DATASET/*'):
        print('CLASS NAME : ', class_name)

        if (class_name.find('WAVE') != -1):
            action_ID = 0
        elif (class_name.find('WALK') != -1):
            action_ID = 1
        elif (class_name.find('IDLE') != -1):
            action_ID = 2
        elif (class_name.find('CAPI') != -1):
            action_ID = 3

        str_class_name = str(class_name)
        for name in glob.glob(class_name + '/*'):
            cap = cv2.VideoCapture(name)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))             
            ehpi_counter = 0

            # print('NAME : ',name)

            a = a + 1
            s = s +1    
            input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
            fps_tracker = FPSTracker(average_over_seconds=1)    
            img_save = np.zeros((32, 15, 3), dtype=np.float32) 
            img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
            i = 0
            j = 0         
            for frame_nr, frame in enumerate(input_provider.get_data()):        
                print(frame_nr)

                humans = []                    
                humans = pose_net.get_humans_from_img_pure(frame,image_size)

                img_save_vecs =  []    
                for human in humans : 
                    feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)

                    # For saving the frames starting from 15th frame in the video
                    # If used, then the dataset will contain normal consecutive 32 frames starting from the beginnig and also the next 32 frames wich starts from 15th frame of the video
                    # The saved clips will overlap for 15 frames
                    # Dataset would have approximately the doubled length
                                    
                    if frame_nr >= 15 :  #### Optional but recommended ####
                        if j < 31 : 
                            img_save_mitte[j,:,:] = feature_vector
                            j = j + 1
                            ehpi_counter = 1
                        elif j == 31 :
                            ehpi_counter = 2
                            img_save_mitte[j,:,:] = feature_vector
                            j = 0
                            ehpi_img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)
                            ehpi_img_save_mitte = img_save_mitte                                            
                            ehpi_img_save_transpose_reshaped_mitte =  ehpi_img_save_mitte.reshape(1,1440) 
                            
                            y_csv_array_mitte = np.zeros((1,2))        
                            if s%10 == 0 :
                                s=s+1                            
                            y_csv_array_mitte[0,0]=action_ID # giving the action ID to all the colums corresponding to x_dataset
                            y_csv_array_mitte[0,1]=s # so that specific_action_ID her action da yeni bir ID alsin diye -- a will be +1 with every different sample
                            
                            with open('/home/xavier1/PrePro/complete_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/home/xavier1/PrePro/complete_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
            
                            if a % 5 == 0 : 
                                with open('/home/xavier1/PrePro/X_test.csv','a') as fd:
                                    np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                                with open('/home/xavier1/PrePro/y_test.csv','a') as fd:
                                    np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                            else :
                                with open('/home/xavier1/PrePro/X_train.csv','a') as fd:
                                    np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                                with open('/home/xavier1/PrePro/y_train.csv','a') as fd:
                                    np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
            
                            img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                    
                    if i < 31 : #### Mandatory ####
                        img_save[i,:,:] = feature_vector
                        i = i + 1
                        ehpi_counter = 1
                    elif i == 31 :
                        ehpi_counter = 2
                        img_save[i,:,:] = feature_vector
                        i = 0
                        ehpi_img_save = np.zeros((32, 15, 3), dtype=np.float32)
                        ehpi_img_save = img_save                                        
                        ehpi_img_save_transpose_reshaped =  ehpi_img_save.reshape(1,1440)               
                        y_csv_array = np.zeros((1,2))

                        if s%10 == 0 :
                            s=s+1                        
                        y_csv_array[0,0]=action_ID # giving the action ID to all the colums corresponding to x_dataset
                        y_csv_array[0,1]=s # so that specific_action_ID her action da yeni bir ID alsin diye -- a will be +1 with every different sample
                        
                        with open('/home/xavier1/PrePro/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/home/xavier1/PrePro/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 

                        if a % 5 == 0 : 
                            with open('/home/xavier1/PrePro/X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                            with open('/home/xavier1/PrePro/y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                                
                        else :
                            with open('/home/xavier1/PrePro//X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                            with open('/home/xavier1/PrePro/y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                     
                                
                        img_save = np.zeros((32, 15, 3), dtype=np.float32)  
                                    
                # saving the video last step :
                # out.write(img)                                                 
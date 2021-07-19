#!/usr/bin/env python3
# Live Test ehpi ofp prod
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

action_save: Dict[str, List[List[float]]] = {}


if __name__ == '__main__':
    setup_application()

    skeleton_type = SkeletonStickman
    image_size = ImageSize(width=1280, height=720)
    heatmap_size = ImageSize(width=64, height=114)
    camera_number = 0
    fps = 30
    buffer_size = 20    
    use_action_recognition = True
    use_quick_n_dirty = False
    
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
    

##########################################################################################################################
#                                                     1. DATASET 
##########################################################################################################################
    
    s = 0    
    a = 0
    action_ID = 0
    # for name in glob.glob('/Users/sk82620/Desktop/DATASET/WAVE_1/*.mp4'):
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/WAVE_1/*.mp4'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )        
        # rest = length % 32               
        ehpi_counter = 0


        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)

            last_humans = image_content_buffer.get_last_humans()
            humans = []
                
            humans = pose_net.get_humans_from_img_pure(frame,image_size)
            # For detecting humans from previous frames 
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans, previous_humans=last_humans)

            human_bbs = [human.bounding_box for human in humans]
            img = get_human_pose_image(frame, humans, min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show) 
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                     
                           
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)  
                               
            # saving the video last step :
            # out.write(img) 
                                      
        fps_tracker.print_fps()        


    a = 0
    action_ID = 1
    for name in glob.glob('/Users/sk82620/Desktop/WALK_1/*.mp4'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        
    
    a = 0
    action_ID = 2
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/IDLE_1/*.mp4'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        
    
    a = 0
    action_ID = 3
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/CAPI_1/*.mp4'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        
    


##########################################################################################################################
#                                                     2. DATASET 
##########################################################################################################################


    a = 0
    action_ID = 0
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/WAVE_2/*.MOV'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                  
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                     
                           
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)  
                      
        fps_tracker.print_fps()        


    a = 0
    action_ID = 1
    for name in glob.glob('/Users/sk82620/Desktop/WALK_2/*.MOV'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        
    
    a = 0
    action_ID = 2
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/IDLE_2/*.MOV'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        
    
    a = 0
    action_ID = 3
    for name in glob.glob('/Users/sk82620/Desktop/DATASET/CAPI_2/*.MOV'):
        cap = cv2.VideoCapture(name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )                      
        ehpi_counter = 0
        
        a = a + 1
        s = s +1
        print("name of the file : "+ name)       
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        j = 0         
        for frame_nr, frame in enumerate(input_provider.get_data()):        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)    
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)    
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)    
            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]    
            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)
    
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []    
            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                               
                if frame_nr >= 15 : 
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
                        
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' ) 
        
                        if a % 5 == 0 : 
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                                    
                        else :
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                                np.savetxt(fd, ehpi_img_save_transpose_reshaped_mitte,delimiter=',',fmt='%1.3f' )         
                            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                                np.savetxt(fd, y_csv_array_mitte,delimiter=',',fmt='%d' )                     
        
                        img_save_mitte = np.zeros((32, 15, 3), dtype=np.float32)                                  
                
                if i < 31 : 
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
                    
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    
                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )     
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )    
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                                             
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)         
        fps_tracker.print_fps()        































    #    if ehpi_counter == 1 :
    #        ehpi_img_save = np.zeros((32, 15, 3), dtype=np.float32)
    #        
    #        r_1 = 32 % rest
    #        r_2 = 32 // rest
    #        tempo = 0            
    #        for mat in range(0,r_2):
    #            for temporal in range(0,rest):
    #                ehpi_img_save[tempo,:,:] = img_save[temporal,:,:]
    #                tempo = tempo + 1
    #                    
    #        for mat in range(0,r_1):
    #            ehpi_img_save[tempo,:,:] = img_save[mat,:,:]
    #            tempo = tempo + 1
    #                                       
    #        # ehpi_img_save = img_save                                        
    #        ehpi_img_save_transpose_reshaped =  ehpi_img_save.reshape(1,1440)                     
    #        y_csv_array = np.zeros((1,2))
    #
    #        if s%10 == 0 :
    #            s=s+1
    #        y_csv_array[0,0]=action_ID # giving the action ID to all the colums corresponding to x_dataset
    #        y_csv_array[0,1]=s # so that specific_action_ID her action da yeni bir ID alsin diye -- a will be +1 with every different sample
    #
    #        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
    #            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 
    #
    #        with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
    #            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 
    #                    
    #        if a % 5 == 0 : 
    #            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
    #                np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 
    #
    #            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
    #                np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
    #                        
    #        else :
    #            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
    #                np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 
    #
    #            with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
    #                np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                    
                        
            # img_save = np.zeros((32, 15, 3), dtype=np.float32)    


# just in case : 
'''
    a = 0
    action_ID = 3
    for name in glob.glob('/Users/sk82620/Desktop/CAPI/*.MOV'):
        ehpi_counter = 0
        ehpi_counter_not_full = 0
        a = a + 1
        s = s +1
        print("name of the file : "+ name)
        
        input_provider = VideoDirProvider(camera_number=name,image_size=image_size, fps=fps)
        
        print(action_ID)
    # input_provider = WebcamProvider(camera_number=1, image_size=image_size, fps=fps)
    # input_provider = ImgDirProvider(img_dir ="/Users/sk82620/Pictures/Camera Roll",image_size=image_size, fps=fps)
        fps_tracker = FPSTracker(average_over_seconds=1)

    # Pose Network
        pose_model = pose_resnet.get_pose_net(pose_resnet_config)

        logger.info('=> loading model from {}'.format(pose_resnet_config.model_state_file))
        pose_model.load_state_dict(torch.load(pose_resnet_config.model_state_file))
        pose_model = pose_model.cuda()
        pose_model.eval()
        pose_net = Pose2DNetResnet(pose_model, skeleton_type)
        pose_tracker = PoseTracker(image_size=image_size, skeleton_type=skeleton_type)

    # Action Network
        action_model = ShuffleNetV2(input_size=32, n_class=26)
        state_dict = torch.load(ehpi_model_state_file)
        action_model.load_state_dict(state_dict)
        action_model.cuda()
        action_model.eval()
        feature_vec_producer = FeatureVecProducerEhpi(image_size,
                                                    get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
        action_net = ActionRecNetEhpi(action_model, feature_vec_producer, image_size)

    # Content Buffer
        image_content_buffer: ImageContentBuffer = ImageContentBuffer(buffer_size=buffer_size)
    
        counts: Dict[str, int] = {}
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        i = 0
        for frame_nr, frame in enumerate(input_provider.get_data()):
        
            print(frame_nr)
            last_humans = image_content_buffer.get_last_humans()
            humans = []
            object_bounding_boxes = []
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)

            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                        previous_humans=last_humans)

            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)
        
      #  for human in humans :
          #  print(human.score_joints)

            human_bbs = [human.bounding_box for human in humans]
            other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]

            img = get_human_pose_image(frame, humans,
                                   min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)

        # bbs_to_draw = [bb for bb in human_data.bbs if bb.label == "person"]
            image_content = ImageContent(humans=humans, objects=human_bbs)
            img_save_vecs =  []

            for human in humans : 
                feature_vector = feature_vec_producer.get_feature_vec(human.skeleton)
                if i < 31 : 
                    img_save[i,:,:] = feature_vector
                    i = i + 1
                elif i == 31 :
                    ehpi_counter = ehpi_counter + 1
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

                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                        np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

                    with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                        np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 

                    if a % 5 == 0 : 
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
                    else :
                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                            np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

                        with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                            np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                     

                        
                    img_save = np.zeros((32, 15, 3), dtype=np.float32)  

        if ehpi_counter == 0 :
            ehpi_counter_not_full = ehpi_counter_not_full + 1
            ehpi_img_save = np.zeros((32, 15, 3), dtype=np.float32)
            ehpi_img_save = img_save
                                        
            ehpi_img_save_transpose_reshaped =  ehpi_img_save.reshape(1,1440) 
                    
            y_csv_array = np.zeros((1,2))
    
            if s%10 == 0 :
                s=s+1
            y_csv_array[0,0]=action_ID # giving the action ID to all the colums corresponding to x_dataset
            y_csv_array[0,1]=s # so that specific_action_ID her action da yeni bir ID alsin diye -- a will be +1 with every different sample

            with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_X_train.csv','a') as fd:
                np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

            with open('/Users/sk82620/Desktop/read_ntu_rgbd/complete_y_train.csv','a') as fd:
                np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' ) 

            if a % 5 == 0 : 
                with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_test.csv','a') as fd:
                    np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

                with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_test.csv','a') as fd:
                    np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )    
                            
            else :
                with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_X_train.csv','a') as fd:
                    np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' ) 

                with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme_y_train.csv','a') as fd:
                    np.savetxt(fd, y_csv_array,delimiter=',',fmt='%d' )                    

                        
            img_save = np.zeros((32, 15, 3), dtype=np.float32)          
            
                
            print(img_save)
                
        fps_tracker.print_fps()
    print(ehpi_counter)
    print(ehpi_counter_not_full)
    
'''
#!/usr/bin/env python3
# Live Test ehpi ofp prod
from collections import deque
from operator import itemgetter
from typing import Dict, List, Tuple
import os

from ehpi_action_recognition.config import pose_resnet_config, pose_visualization_config, ehpi_model_state_file_5, ehpi_model_custom_4_SGD, ehpi_model_custom_4_ADAM

import cv2
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
from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet

from nobos_commons.data_structures.human import Human
from nobos_commons.tools.skeleton_converters.skeleton_converter_coco_to_stickman import SkeletonConverterCocoToStickman

action_save: Dict[str, List[List[float]]] = {}


# IMPORTING ROS MODULE
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import UInt16
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import Bool
# to send action_ID and human_ID fancy in STRING ( wouldn't be that useful to capture information )

def talker(action_name:str):
    pub = rospy.Publisher('gesture', String, queue_size=10)  
    # rate = rospy.Rate(10) # 10hz
    
    gesture_name = action_name
    rospy.loginfo(gesture_name)
    pub.publish(gesture_name)
    
def hand_direction(id:int, right:int, left:int): 
    pub_int = rospy.Publisher('direction_int', Int32, queue_size=10)
    # rospy.init_node('direction_integer', anonymous=True)
    
    id_to_pub = 1000
    right_to_pub = 20
    left_to_pub = 30
    
    id_to_pub = id_to_pub + id
    right_to_pub = right_to_pub + right
    left_to_pub = left_to_pub + left
    
    direction = int(str(id_to_pub) + str(right_to_pub) + str(left_to_pub))
    rospy.loginfo(direction)
    pub_int.publish(direction)  
        
def direction(right:int, left:int): 
    pub_int = rospy.Publisher('gesture_control', UInt16, queue_size=10)
    # rospy.init_node('direction', anonymous=True)
    
    if right == 1 : # LINKS
        direction = 1
    elif left == 1 : # RECHTS 
        direction = 3
    elif right == 2 and left == 2 : 
        direction = 2
        
    # rospy.loginfo(direction)
    pub_int.publish(direction)  

def x_coordinates(hip_x:int): 
    pub_int = rospy.Publisher('human_stalker', UInt16, queue_size=10)       
    # x_coord = int(hip_x)       
    # rospy.loginfo(hip_x)
    pub_int.publish(hip_x)  
    
def spine_len(spine_length:int): 
    pub_int = rospy.Publisher('spine_length', UInt16, queue_size=10)
    # rospy.loginfo(spine_length)
    pub_int.publish(spine_length) 
            
def human_detected(detected:bool): 
    pub_int = rospy.Publisher('human_detected', Bool, queue_size=10)      
    # rospy.loginfo(detected)
    pub_int.publish(detected) 
    
def human_logged_in(logged_in:bool): 
    pub_int = rospy.Publisher('human_logged_in', Bool, queue_size=10)      
    # rospy.loginfo(logged_in)
    pub_int.publish(logged_in)     
 


def get_stabelized_action_recognition(human_id: str, action_probabilities: np.ndarray,joints):
    queue_size = 20
    if human_id not in action_save:
        action_save[human_id] = []
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))
        action_save[human_id].append(deque([0] * queue_size, maxlen=queue_size))

        
    argmax = np.argmax(action_probabilities)
    for i in range(0, 4):
        if i == argmax:
            action_save[human_id][i].append(1)
        else:
            action_save[human_id][i].append(0)
    
    # print(action_save[human_id][0]) ---- bu garip deque lu 1 ve 0 li halini veriyor
    # print(sum(action_save[human_id][0]) / queue_size) ---- bu hali olasiliklarini veriyor direk 

    WALK = float("{:.2f}".format((sum(action_save[human_id][1]) / queue_size)))
    
    if (joints[0][0] < 0.50)and(joints[1][0] < 0.50)and(joints[2][0] < 0.50)and(joints[3][0] < 0.50):
        WALK = 0
    
    # OTHER ADJUSMENTS :     
    #HAND_WAVE = float("{:.2f}".format((sum(action_save[human_id][6]) / queue_size)))
    
    return [sum(action_save[human_id][0]) / queue_size,
            WALK,
            sum(action_save[human_id][2]) / queue_size,
            sum(action_save[human_id][3]) / queue_size]


def argmax(items):
    index, element = max(enumerate(items), key=itemgetter(1))
    #print("index :" + str(index))
    #print("element :" +str(element))
    
    # if maximum action score is less than 0.3 -- will be automatically IDLE 
    if element < 0.11 : 
        index = 2
        element = 1
    return index, element

if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    setup_application()
    # Settings
    skeleton_type = SkeletonStickman
    image_size = ImageSize(width=640, height=480)
    # image_size = ImageSize(width=215, height=601)
    heatmap_size = ImageSize(width=64, height=114)
    camera_number = 0
    fps = 30
    buffer_size = 20

    action_names = [
    Action.HAND_WAVE.name,
    Action.WALK.name,    
    Action.IDLE.name,
    Action.CAPITULATE.name]
   
    use_action_recognition = False
    use_quick_n_dirty = False
    frame_nr = 1
    hum = 1000
    dogru_id = 0  
    
        
    # for saving video 
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('/Users/sk82620/Pictures/Camera Roll/uni_4_proc_mid_speed.mp4', fourcc, 25.0, (1280, 720))

    # INPUT PROVIDER
    # input_provider = VideoDirProvider(camera_number='/home/xavier1/catkin_ws/src/gui/src/video.MOV',image_size=image_size, fps=fps)
    input_provider = WebcamProvider(camera_number=0, image_size=image_size, fps=fps)
    # input_provider = ImgDirProvider(img_dir ="/Users/sk82620/Pictures/Camera Roll/yolo",image_size=image_size, fps=fps)
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
    action_model = ShuffleNetV2(input_size=32, n_class=4)
    state_dict = torch.load(ehpi_model_custom_4_SGD)
    action_model.load_state_dict(state_dict)
    action_model.cuda()
    action_model.eval()
    feature_vec_producer = FeatureVecProducerEhpi(image_size, get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
    action_net = ActionRecNetEhpi(action_model, feature_vec_producer, image_size)

    # Content Buffer
    image_content_buffer: ImageContentBuffer = ImageContentBuffer(buffer_size=buffer_size)
    time.sleep(1)
    counts: Dict[str, int] = {}

    human_single: List[Human] = []

    print('CUDA')
    print(torch.cuda.is_available())

    for frame_nr, frame in enumerate(input_provider.get_data()):
        
        last_humans = image_content_buffer.get_last_humans()
        humans = []
        object_bounding_boxes = []

        if len(human_single) == 0:
            # For detecting new humans in each frame
            if not use_quick_n_dirty or last_humans is None or len(last_humans) == 0:
                object_bounding_boxes = pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = pose_net.get_humans_from_bbs(frame, human_bbs)

            # For detecting humans from previous frames 
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans, previous_humans=last_humans)
            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)
            
            # extension for better merge !!!! 
            humans = pose_tracker.get_humans_merge_in_end(frame, detected_humans=humans)
            human_bbs = [human.bounding_box for human in humans]

            tmp_h1 = 0
            tmp_h2 = 0
            humans_heights = []

            for human in humans:
                # print(human.bounding_box.height)
                if human.bounding_box.height > tmp_h1 and human.bounding_box.height > tmp_h2: 
                    if tmp_h1 <= tmp_h2:
                        tmp_h1 = human.bounding_box.height
                        human_tmp1 = human
                        human_bb1 = human.bounding_box
                    else:
                        tmp_h2 = human.bounding_box.height
                        human_tmp2 = human
                        human_bb2 = human.bounding_box

                if (human.bounding_box.height < tmp_h1 and human.bounding_box.height > tmp_h2): 
                    tmp_h2 = human.bounding_box.height
                    human_tmp2 = human
                    human_bb2 = human.bounding_box

                if (human.bounding_box.height > tmp_h1 and human.bounding_box.height < tmp_h2): 
                    tmp_h1 = human.bounding_box.height
                    human_tmp1 = human
                    human_bb1 = human.bounding_box

            if tmp_h1 != 0 and tmp_h2 != 0:
                humans_heights = [human_tmp1, human_tmp2]
                human_bbs = [human_bb1, human_bb2]

            #if len(humans_heights) == 2:
             #   print('burdaim humansss')
              #  humans = humans_heights

            img = get_human_pose_image(frame, humans, min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)

            # bbs_to_draw = [bb for bb in human_data.bbs if bb.label == "person"]
            image_content = ImageContent(humans=humans, objects=human_bbs)
            image_content_buffer.add(image_content)
            actions: Dict[str, Tuple[Action, float]] = {}
        else:
            # For detecting humans from previous frames 
            humans, undetected_humans = pose_tracker.get_humans_by_tracking(frame, detected_humans=humans, previous_humans=human_single)

            redetected_humans = pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
            humans.extend(redetected_humans)
            
            # extension for better merge !!!! 
            humans = pose_tracker.get_humans_merge_in_end(frame, detected_humans=humans)
            human_bbs = [human.bounding_box for human in humans]

            img = get_human_pose_image(frame, humans, min_limb_score_to_show=pose_visualization_config.min_limb_score_to_show)

            # bbs_to_draw = [bb for bb in human_data.bbs if bb.label == "person"]
            image_content = ImageContent(humans=humans, objects=human_bbs)
            image_content_buffer.add(image_content)
            actions: Dict[str, Tuple[Action, float]] = {}


        right = 0
        left = 0
        spine_length = 0
        hip_x = 0       
        detected = False
        logged = False
    
        if len(humans) != 0 :
            detected = True
            human_detected(detected) 
        else :
            detected = False
            human_detected(detected)

##########################################################################################################################            
        # IF ACTION RECOGNITION IS NOT WANTED BUT JUST THE JOINT INFORMATION :
##########################################################################################################################
        
        if use_action_recognition == False : 
            for human in humans:    
                right, left,spine_length, hip_x = feature_vec_producer.get_direction(human.skeleton)
                # self.direction(right, left)
                x_coordinates(hip_x)
                spine_len(spine_length)

                img = draw_bb(img, human.bounding_box, actions[human.uid][0].name if human.uid in actions else "NONE", " {0:.2f}".format(human.score), human.uid)
                
##########################################################################################################################    
        # FOR ACTION RECOGNITION : 
##########################################################################################################################
        
        if use_action_recognition == True:
            action_results,joint_scores, right, left, spine_length, neck_x = action_net.get_actions(humans, frame_nr)
            for human_id, action_logits in action_results.items():
                
                action_probabilities = softmax(action_logits)
                actions[human_id] = []
                predictions = get_stabelized_action_recognition(human_id, action_probabilities,joint_scores)
                            
                pred_label, probability = argmax(predictions)
                actions[human_id] = (Action[action_names[pred_label]], probability) 

                #### 

                human_single: List[Human] = []   
                print('humans')
                print(len(humans))

                for human in humans :             
                    if human.uid in actions and actions[human.uid][0].name == "HAND_WAVE" and hum == 1000 : 
                        # must always have an action, NONE wouldn't be accepted -- but in any case if it becomes NONE, means that it will change ID !
                        hum = int(human.uid)   

                    if human.uid in actions and hum == int(human.uid) : # generally to check if the same ID person still in the picture to be seen is or not      
                        # Use only one person detection after a person is logged in -- save single person in the list to be given bask as 'last human'
                        use_quick_n_dirty = True                 
                        human_single.append(human)  

                        hip_x = 2                            
                        right, left, spine_length, hip_x = feature_vec_producer.get_direction(human.skeleton)
                        if hip_x != 2:
                            logged = True
                            x_coordinates(int(hip_x))
                            spine_len(int(spine_length))
                            human_logged_in(logged)                    
                            # self.all_together(detected, int(hip_x), int(spine_length))

                        dogru_id = 1 
                        img = draw_bb_stalker(img, human.bounding_box, actions[human.uid][0].name if human.uid in actions else "NONE", " {0:.2f}".format(human.score), human.uid)                

                        if actions[human.uid][0].name == "CAPITULATE" :
                            use_quick_n_dirty = False
                            hum = 1000   

                    else :
                        img = draw_bb(img, human.bounding_box, actions[human.uid][0].name if human.uid in actions else "NONE", " {0:.2f}".format(human.score), human.uid)

                if logged == False : 
                    human_logged_in(logged)

                if all([human.uid in actions and hum != int(human.uid) for human in humans]):
                    dogru_id = 0 

                if dogru_id == 0 : 
                    hum = 1000 
                #bbs_to_draw = [bb for bb in image_content.objects if bb.label != "person"]  
                

        # --------- Draw objects
        #bbs_to_draw = [bb for bb in image_content.objects if bb.label != "person"]
        #list(map(lambda bb: draw_bb(img, bb), bbs_to_draw))

        #list(map(lambda human: draw_bb(img, human.bounding_box, actions[human.uid][0].name if human.uid in actions else "NONE", " {0:.2f}".format(human.score), human.uid), image_content.humans))       
        
        # saving the video last step :
        # out.write(img)

        img = cv2.resize(img, (640, 480))
        cv2.imshow('webcam', img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        fps_tracker.print_fps()
        

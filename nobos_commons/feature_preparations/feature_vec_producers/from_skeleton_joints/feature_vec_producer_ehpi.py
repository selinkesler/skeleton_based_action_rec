# TODO: This will be the feature vec representing 1 column of a ehpi image
# TODO: 3 (or 6?) rows should correspond to an body part like left arm, right arm etc. for a corresponding kernel size in
# the network
# TODO: Use body parts: left arm, right arm, left leg, righ leg, (neck, lhip, rhip), (head, neck, hip), (hip, lfoot, rfoot),
# TODO: (hip, lhand, rhand), (neck, leye, reye)
# TODO: How to represend a spatial pose (n rows)? How To represen temporal pose (n columns)?
# TODO: Try to normalize everything by the humans height
from typing import Callable

import numpy as np
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import \
    get_joints_default
from nobos_commons.utils.human_surveyor import HumanSurveyor


class FeatureVecProducerEhpi(object):
    def __init__(self, image_size: ImageSize, human_surveyor: HumanSurveyor = HumanSurveyor(),
                 get_joints_func: Callable = lambda skeleton: get_joints_default(skeleton),
                 skeleton: SkeletonStickman = SkeletonStickman()):
        self.__image_size = image_size
        self.human_surveyor = human_surveyor
        self.get_joints_func: Callable = get_joints_func
        joints = self.get_joints_func(skeleton)
        self.num_joints = len(joints)
        # We have 4 features for each combination of 2 joints and 3 features for each combination of 3 joints
        self.feature_vec_length = int(3 * self.num_joints)

    def get_feature_vec(self, skeleton: SkeletonStickman) -> np.ndarray:
        """
        Returns the (unnormalized) feature vec
        :param skeleton:
        :return:
        """
        # human_height = self.human_surveyor.get_human_height(skeleton.limbs)
        joints = self.get_joints_func(skeleton)
        # print(joints)
        joint_score_array = np.zeros((4, 1), dtype=np.float32)
        feature_vec = np.zeros((len(joints), 3), dtype=np.float32)
        for idx, joint in enumerate(joints):
            # my additions for getting rid of wrong leg data -- avoid constant walking/ kicking
            # if ((joint.num == 12)or(joint.num == 9)or(joint.num == 13)or(joint.num == 10))and joint.score < 0.55 :
              #  joint.score = 0
               # print(str(joint.num) + "is gone")
            if joint.score < 0.4 :
                feature_vec[idx][0] = 0
                feature_vec[idx][1] = 0
                feature_vec[idx][2] = 0
            else:
                feature_vec[idx][0] = joint.x
                feature_vec[idx][1] = joint.y
                # feature_vec[idx][2] = joint.score
        # print("feature vec :")      
        # print(feature_vec)

        return feature_vec
        
        

    def get_joint_scores(self, skeleton: SkeletonStickman) -> np.ndarray:
        joints = self.get_joints_func(skeleton)
        joint_score_array = np.zeros((4, 1), dtype=np.float32)
        for idx, joint in enumerate(joints):
            if (joint.num == 12):
                joint_score_array[0][0]=joint.score
            if (joint.num == 13):
                joint_score_array[1][0]=joint.score 
            if (joint.num == 9):
                joint_score_array[2][0]=joint.score 
            if (joint.num == 10):
                joint_score_array[3][0]=joint.score 
        return joint_score_array
        
        
    def get_direction(self, skeleton: SkeletonStickman) -> np.ndarray:
        joints = self.get_joints_func(skeleton)
        # joint_score_array = np.zeros((4, 1), dtype=np.float32)
        right = 0
        left = 0 
        body_center_y = 0
        left_wrist_y = 0
        right_wrist_y = 0
        neck_y = 0
        hip_x = 2000
        hip_y = 0
        
        
        for idx, joint in enumerate(joints):
            if (joint.num == 4):
                right_wrist_y = joint.y
            if (joint.num == 7):
                left_wrist_y = joint.y
            if (joint.num == 1):
                neck_y = joint.y
            if (joint.num == 18):
                hip_y = joint.y
                hip_x = joint.x
                
        spine_length =  hip_y - neck_y
        body_center_y = (neck_y + hip_y)/2
        
        # print("spine lenght:" +str(spine_length))

                
        if  right_wrist_y < body_center_y and left_wrist_y > body_center_y :
            right = 1
            # print("RIGHT")
            
        elif right_wrist_y > body_center_y and left_wrist_y < body_center_y :
            left = 1
            # print("LEFT")   
            
        elif right_wrist_y < body_center_y and left_wrist_y < body_center_y :
        
            if right_wrist_y < left_wrist_y :
                right = 1
                #print("RIGHT")
            else :
                left = 1
                #print("LEFT")
                
        elif right_wrist_y > body_center_y and left_wrist_y > body_center_y :
            right = 2
            left = 2
            
        return right, left, spine_length, hip_x


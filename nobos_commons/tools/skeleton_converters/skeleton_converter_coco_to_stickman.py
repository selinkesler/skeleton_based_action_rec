from nobos_commons.data_structures.skeletons.joint_2d import Joint2D
from nobos_commons.data_structures.skeletons.skeleton_coco import SkeletonCoco
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.tools.skeleton_converters.skeleton_converter_base import SkeletonConverter
from nobos_commons.utils.joint_helper import get_middle_joint,get_blick_joint,get_forward_joint,get_blick_length_joint,get_procent,get_new_refrence_point,get_left_right


class SkeletonConverterCocoToStickman(SkeletonConverter):
    def get_converted_skeleton(self, skeleton_coco: SkeletonCoco) -> SkeletonStickman:
        skeleton_stickman: SkeletonStickman = self._get_skeleton_from_joints(skeleton_coco)
        self._set_calculated_joints(skeleton_stickman)
        return skeleton_stickman

        
    def dene(self, skeleton_coco: SkeletonCoco) -> SkeletonStickman:
        a=1
        return 1
    # Private methods

    def _get_skeleton_from_joints(self, skeleton_coco: SkeletonCoco) -> SkeletonStickman:
        skeleton_stickman: SkeletonStickman = SkeletonStickman()
        for joint in skeleton_coco.joints:
            skeleton_stickman.joints[joint.name].copy_from(joint, allow_different_num=True)
        return skeleton_stickman

    def _set_calculated_joints(self, skeleton_stickman: SkeletonStickman):
        calculated_neck: Joint2D = get_middle_joint(joint_a=skeleton_stickman.joints.left_shoulder,
                                                    joint_b=skeleton_stickman.joints.right_shoulder)

        calculated_hip_center: Joint2D = get_middle_joint(joint_a=skeleton_stickman.joints.left_hip,
                                                          joint_b=skeleton_stickman.joints.right_hip)
                                                          
        if calculated_neck is not None:
            skeleton_stickman.joints.neck.copy_from(calculated_neck)

        if calculated_hip_center is not None:
            skeleton_stickman.joints.hip_center.copy_from(calculated_hip_center)      


            
        # MY ADDITIONS : 
        calculated_ear_middle: Joint2D = get_middle_joint(joint_a=skeleton_stickman.joints.left_ear,
                                                          joint_b=skeleton_stickman.joints.right_ear)
                                                          
        calculated_eye_middle: Joint2D = get_middle_joint(joint_a=skeleton_stickman.joints.left_eye,
                                                          joint_b=skeleton_stickman.joints.right_eye)                                                         

        if calculated_ear_middle is not None:
            skeleton_stickman.joints.ear_center.copy_from(calculated_ear_middle)

        if calculated_eye_middle is not None:
            skeleton_stickman.joints.eye_center.copy_from(calculated_eye_middle)            
            
        '''    
        calculated_eye_blick: Joint2D = get_blick_joint(joint_ear_center=skeleton_stickman.joints.ear_center,
                                                          joint_Reye=skeleton_stickman.joints.right_eye, joint_Leye=skeleton_stickman.joints.left_eye)

        if calculated_eye_blick is not None:
            skeleton_stickman.joints.eye_blick.copy_from(calculated_eye_blick) 
        '''

        
        calculated_refrence_joint : Joint2D = get_new_refrence_point(joint_ear_center = skeleton_stickman.joints.ear_center, joint_eye_center = skeleton_stickman.joints.eye_center )
        if calculated_refrence_joint is not None:
            skeleton_stickman.joints.eye_blick.copy_from(calculated_refrence_joint)    
            
            
            
        calculated_forward_eye: Joint2D = get_blick_length_joint(joint_eye_center=skeleton_stickman.joints.eye_center,
                                                          joint_blick=skeleton_stickman.joints.eye_blick) 
        if calculated_forward_eye is not None:
            skeleton_stickman.joints.eye_forward.copy_from(calculated_forward_eye) 

        procent = get_procent(joint_eye_center = skeleton_stickman.joints.eye_center, joint_blick=skeleton_stickman.joints.eye_blick, joint_neck = skeleton_stickman.joints.neck, joint_hip_center = skeleton_stickman.joints.hip_center)
        
        
        
        # FOR PROCENT of HOW MUCH LEFT OR RIGHT is the person looking at ?? 
        
        
    #    if procent < 0 :
    #        procent = abs(procent)
    #        
    #        print("procent = LEFT {}".format(procent)) 
    #    else: 
    #        print("procent = RIGHT {}".format(procent))
            

        
        calculated_middle_belly: Joint2D = get_middle_joint(joint_a=skeleton_stickman.joints.hip_center,
                                                    joint_b=skeleton_stickman.joints.neck)
                                                    
        if calculated_middle_belly is not None:
            skeleton_stickman.joints.body_center.copy_from(calculated_middle_belly) 
            
            
        # print("CALIS")
        # print(skeleton_stickman.joints.body_center)
            
       # y_left,y_right = get_left_right(joint_belly_center=skeleton_stickman.joints.body_center, joint_right_wrist=skeleton_stickman.joints.right_wrist,joint_left_wrist=skeleton_stickman.joints.left_wrist )
        
                                                    
        
    def get_infomation(self, skeleton_stickman: SkeletonStickman):
        y_left,y_right = get_left_right(joint_belly_center=skeleton_stickman.joints.body_center,
                                                    joint_right_wrist=skeleton_stickman.joints.right_wrist,joint_left_wrist=skeleton_stickman.joints.left_wrist )

        return y_left, y_right
        
        
    #
    #
    # _coco_stickman_mapping = {
    #     'Nose': 'nose',
    #     'LEye': 'left_eye',
    #     'REye': 'right_eye',
    #     'LEar': 'left_ear',
    #     'REar': 'right_ear',
    #     'LShoulder': 'left_shoulder',
    #     'RShoulder': 'right_shoulder',
    #     'LElbow': 'left_elbow',
    #     'RElbow': 'right_elbow',
    #     'LWrist': 'left_wrist',
    #     'RWrist': 'right_wrist',
    #     'LHip': 'left_hip',
    #     'RHip': 'right_hip',
    #     'LKnee': 'left_knee',
    #     'RKnee': 'right_knee',
    #     'LAnkle': 'left_ankle',
    #     'RAnkle': 'right_ankle'
    # }

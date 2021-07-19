import math
from typing import List

from nobos_commons.data_structures.geometry import Triangle
from nobos_commons.data_structures.skeletons.joint_2d import Joint2D
from nobos_commons.data_structures.skeletons.joint_visibility import JointVisibility


def get_euclidean_distance_joint2d(joint_a: Joint2D, joint_b: Joint2D) -> float:
    return math.hypot(joint_b.x - joint_a.x, joint_b.y - joint_a.y)


def get_euclidean_distance_joint_lists(joints_a: List[Joint2D], joints_b: List[Joint2D], min_joint_score: float = 0.0) -> List[float]:
    """
    Returns the distance of the correspondiong joints of two lists. The lists must have the same length
    :param min_joint_score: The minimum score for both joints to be included in the distance check
    :param joints_a:
    :param joints_b:
    :return: List of floats for each joint_id in the lists with the euclidean distance
    """
    assert len(joints_a) == len(joints_b)
    joint_distances = []
    for joint_id, joint_tuple in enumerate(zip(joints_a, joints_b)):
        joint_a, joint_b = joint_tuple
        if joint_a.score >= min_joint_score and joint_b.score >= min_joint_score:
            joint_distances.append(get_euclidean_distance_joint2d(joint_a, joint_b))
    return joint_distances


def get_distances(joint_a: Joint2D, joint_b: Joint2D) -> (float, float, float):
    """
    Calculates the distances between the x and y coordinates as well as the euclidean distance between the joints.
    :param joint_a: 2D joint from
    :param joint_b: 2D joint to
    :return: (
    distance between the joint's x coordinates,
    distance between the joint's x coordinates,
    euclidean distance between the joints
    )
    """
    distance_x = abs(joint_a.x - joint_b.x)
    distance_y = abs(joint_a.y - joint_b.y)
    euclidean_distance = get_euclidean_distance_joint2d(joint_a, joint_b)
    return distance_x, distance_y, euclidean_distance


def get_angle_rad_between_joints(joint_a: Joint2D, joint_b: Joint2D) -> float:
    """
    Returns the angle between two joints in radians. Result between -pi and +pi
    """
    return math.atan2(joint_a.y - joint_b.y, joint_a.x - joint_b.x)


def get_triangle_from_joints(joint_a: Joint2D, joint_b: Joint2D, joint_c: Joint2D) -> Triangle:
    """
    Returns alpha, beta and gamma in a triangle formed by three joints (in radians).
    length_a = length_line c->b
    length_b = length_line c->a
    length_c = length_line a->b
    alpha = angle between joint_b and joint_c
    beta = angle between joint_a and joint_c
    gamma = angle between joint_a and joint_b
    cos alpha = (b^2 + c^2 - a^2) / (2 * b * c)
    cos beta = (a^2 + c^2 - b^2) / (2 * a * c)
    gamma = pi - alpha - beta
    :param joint_a: 2D joint
    :param joint_b: 2D joint
    :param joint_c: 2D joint
    :return: (alpha_rad, beta_rad, gamma_rad)
    """
    length_a = get_euclidean_distance_joint2d(joint_c, joint_b)
    length_b = get_euclidean_distance_joint2d(joint_c, joint_a)
    length_c = get_euclidean_distance_joint2d(joint_a, joint_b)
    # Note: Round to prevent round errors on later decimals on extremes (1.0, -1.0)
    # TODO: How to handle 0 distance correctly?
    if length_a == 0 or length_b == 0 or length_c == 0:
        return Triangle(0, 0, 0, 0, 0, 0)
    cos_alpha = round((((length_b ** 2) + (length_c ** 2) - (length_a ** 2)) / (2 * length_b * length_c)), 2)
    alpha_rad = math.acos(cos_alpha)
    cos_beta = round((((length_a ** 2) + (length_c ** 2) - (length_b ** 2)) / (2 * length_a * length_c)), 2)
    beta_rad = math.acos(cos_beta)
    gamma_rad = math.pi - alpha_rad - beta_rad
    return Triangle(length_a, length_b, length_c, alpha_rad, beta_rad, gamma_rad)


def get_middle_joint(joint_a: Joint2D, joint_b: Joint2D) -> Joint2D:
    """
    Returns a joint which is in the middle of the two input joints. The visibility and score is estimated by the
    visibility and score of the two surrounding joints.
    :param joint_a: Surrounding joint one
    :param joint_b: Surrounding joint two
    :return: Joint in the middle of joint_a and joint_b
    """
    if not joint_a.is_set or not joint_b.is_set:
        return None
    visibility: JointVisibility
    if joint_a.visibility == JointVisibility.VISIBLE and joint_b.visibility == JointVisibility.VISIBLE:
        visibility = JointVisibility.VISIBLE
    elif joint_a.visibility == JointVisibility.INVISIBLE or joint_b.visibility == JointVisibility.INVISIBLE:
        visibility = JointVisibility.INVISIBLE
    elif joint_a.visibility == JointVisibility.ABSENT or joint_b.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT

    return Joint2D(
        x=((joint_a.x + joint_b.x) / 2),
        y=((joint_a.y + joint_b.y) / 2),
        score=(joint_a.score + joint_b.score) / 2,
        visibility=visibility
    )

def get_blick_joint(joint_ear_center: Joint2D, joint_Reye: Joint2D, joint_Leye: Joint2D ) -> Joint2D:
    if not joint_ear_center.is_set or not joint_Reye.is_set or not joint_Leye.is_set:
        return None
    visibility: JointVisibility
    if joint_ear_center.visibility == JointVisibility.VISIBLE and (joint_Reye.visibility == JointVisibility.VISIBLE or joint_Leye.visibility == JointVisibility.VISIBLE):
        visibility = JointVisibility.VISIBLE
    elif joint_ear_center.visibility == JointVisibility.INVISIBLE or joint_Reye.visibility == JointVisibility.INVISIBLE:
        visibility = JointVisibility.INVISIBLE
    elif joint_ear_center.visibility == JointVisibility.ABSENT or joint_Reye.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT
  
        
    return Joint2D(
        x= joint_ear_center.x,
        y=((joint_Leye.y + joint_Reye.y) / 2),
        score=(joint_ear_center.score + joint_Reye.score + joint_Leye.score ) / 3,
        visibility=visibility
    )  
    
def get_forward_joint(joint_eye_center: Joint2D) -> Joint2D:
    if not joint_eye_center.is_set:
        return None
    visibility: JointVisibility
    if joint_eye_center.visibility == JointVisibility.VISIBLE:
        visibility = JointVisibility.VISIBLE
    elif joint_eye_center.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT
        
        
    return Joint2D(
        x= joint_eye_center.x + 30,
        y=joint_eye_center.y,
        score=joint_eye_center.score,
        visibility=visibility
    )  
    
    
def get_blick_length_joint(joint_eye_center: Joint2D, joint_blick: Joint2D) -> Joint2D:
    if not joint_eye_center.is_set or not joint_blick.is_set:
        return None
    visibility: JointVisibility
    if joint_eye_center.visibility == JointVisibility.VISIBLE and joint_blick.visibility == JointVisibility.VISIBLE:
        visibility = JointVisibility.VISIBLE
    elif joint_eye_center.visibility == JointVisibility.INVISIBLE or joint_blick.visibility == JointVisibility.INVISIBLE:
        visibility = JointVisibility.INVISIBLE
    elif joint_eye_center.visibility == JointVisibility.ABSENT or joint_blick.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT
        
    
    length_x = joint_eye_center.x - joint_blick.x
    lengt_y = joint_eye_center.y - joint_blick.y

    
    return Joint2D(
        x= joint_eye_center.x + length_x,
        y=joint_eye_center.y + lengt_y,
        score=(joint_eye_center.score + joint_blick.score) / 2,
        visibility=visibility
    )  
    
    
def get_procent(joint_eye_center: Joint2D, joint_blick: Joint2D, joint_neck: Joint2D, joint_hip_center: Joint2D):
    if not joint_eye_center.is_set or not joint_blick.is_set or not joint_neck.is_set or not joint_hip_center.is_set:
        return None
        
    
    length_eye = joint_eye_center.x - joint_blick.x
    length_upper_arm = math.hypot(joint_neck.x - joint_hip_center.x, joint_neck.y - joint_hip_center.y)
    length_upper_arm_procent = length_upper_arm/4
    procent = (length_eye * 100)/ length_upper_arm_procent
    
    return procent

    
    return Joint2D(
        x= joint_eye_center.x + length,
        y=joint_eye_center.y,
        score=(joint_eye_center.score + joint_blick.score) / 2,
        visibility=visibility
    ) 
    
    
    
def get_new_refrence_point(joint_ear_center: Joint2D, joint_eye_center: Joint2D) -> Joint2D:
    if not joint_ear_center.is_set or not joint_eye_center.is_set:
        return None
    visibility: JointVisibility
    if joint_eye_center.visibility == JointVisibility.VISIBLE and joint_ear_center.visibility == JointVisibility.VISIBLE:
        visibility = JointVisibility.VISIBLE
    elif joint_eye_center.visibility == JointVisibility.INVISIBLE or joint_ear_center.visibility == JointVisibility.INVISIBLE:
        visibility = JointVisibility.INVISIBLE
    elif joint_eye_center.visibility == JointVisibility.ABSENT or joint_ear_center.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT
        
    euc_distance = math.hypot(joint_ear_center.x - joint_eye_center.x, joint_ear_center.y - joint_eye_center.y)
    b = abs(joint_eye_center.x - joint_ear_center.x)
    a_2 = (euc_distance * euc_distance) - (b*b)
    # a = math.sqrt(a_2)
    a = euc_distance/4
    
    
    # print("eye center x:")
    # print(joint_eye_center.x)
    # print("ear center x:")
    # print(joint_ear_center.x)
    
    
    x_distance = abs(joint_ear_center.x - joint_eye_center.x)
    if x_distance < 4:
        return Joint2D(
            x= joint_ear_center.x,
            y= joint_eye_center.y,
            score=(joint_eye_center.score + joint_ear_center.score) / 2,
            visibility=visibility
        )  
    else :
        return Joint2D(
            x= joint_ear_center.x,
            y= joint_ear_center.y - a,
            score=(joint_eye_center.score + joint_ear_center.score) / 2,
            visibility=visibility
        )  
        
        
def get_left_right(joint_belly_center: Joint2D, joint_right_wrist: Joint2D, joint_left_wrist: Joint2D):
    # print('joint_belly_center : ', joint_belly_center)
    # print('joint_right_wrist : ', joint_right_wrist)
    # print('joint_left_wrist : ', joint_left_wrist)


    if not joint_belly_center.is_set or not joint_right_wrist.is_set or not joint_left_wrist.is_set:
        return None
    visibility: JointVisibility
    if joint_belly_center.visibility == JointVisibility.VISIBLE and joint_right_wrist.visibility == JointVisibility.VISIBLE:
        visibility = JointVisibility.VISIBLE
    elif joint_belly_center.visibility == JointVisibility.INVISIBLE or joint_right_wrist.visibility == JointVisibility.INVISIBLE:
        visibility = JointVisibility.INVISIBLE
    elif joint_belly_center.visibility == JointVisibility.ABSENT or joint_right_wrist.visibility == JointVisibility.ABSENT:
        visibility = JointVisibility.ABSENT
        
    y_right = 0
    y_left = 0
    if joint_right_wrist.y > joint_belly_center.y and joint_left_wrist.y < joint_belly_center.y : 
        y_right = 1
        # print("RIGHT")
    elif joint_left_wrist.y > joint_belly_center.y and joint_right_wrist.y < joint_belly_center.y :
        y_left = 1
        # print("LEFT")
    elif  joint_left_wrist.y > joint_belly_center.y and joint_right_wrist.y > joint_belly_center.y:
        if joint_right_wrist.y > joint_left_wrist.y:
            y_right = 1
            # print("RIGHT")
        else :
            y_left = 1 
            # print("LEFT")
    # else : 
        # print("HAND DOWN")

    
    return y_left,y_right
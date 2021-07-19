from typing import List, Dict

import cv2
import numpy as np

from nobos_commons.data_structures.color import Color, Colors
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.skeletons.joint_2d import Joint2D
from nobos_commons.data_structures.skeletons.limb_2d import Limb2D
from nobos_commons.data_structures.skeletons.skeleton_base import SkeletonBase
from PIL import Image, ImageDraw, ImageFont

# Display and Save methods
from nobos_commons.utils.visualization_helper import limb_should_be_displayed


def display_humans(img: np.ndarray, humans: List[Human], wait_for_ms: int = 0, min_limb_score_to_show: float = 0.4):
    """
    Visualizes all human skeletons and straying joints / limbs in the image and displays the image.
    :param img: The original image
    :param humans: The human content in the image
    :param min_limb_score_to_show: The minimum score of limbs to be displayed
    :param wait_for_ms: The time for which the image should be displayed, if zero wait for keypress
    :return: The image with the visualized humans and straying joints / limbs
    """
    img = get_human_pose_image(img, humans, min_limb_score_to_show)
    cv2.imshow("human_pose", img)
    cv2.waitKey(wait_for_ms)


def save_humans_img(img: np.ndarray, humans: List[Human], file_path="human_pose.png", min_limb_score_to_show: float = 0.4):
    """
    Visualizes all human skeletons and straying joints / limbs in the image and saves the image to the given path.
    :param img: The original image
    :param humans: The human content in the image
    :param file_path: The path in which the image with the visualized content should be saved.
    :param min_limb_score_to_show: The minimum score of limbs to be displayed
    :return: The image with the visualized humans and straying joints / limbs
    """
    img = get_human_pose_image(img, humans, min_limb_score_to_show)
    cv2.imwrite(file_path, img)


# Human Poses


def get_human_pose_image(img: np.ndarray, humans: List[Human], min_limb_score_to_show: float = 0.8):
    """
    Visualizes all human skeletons and straying joints / limbs in the image and returns it.
    :param img: The original image
    :param humans: The human content in the image
    :param min_limb_score_to_show: The minimum score of limbs to be displayed
    :return: The image with the visualized humans and straying joints / limbs
    """
    for human in humans:
        img = get_visualized_skeleton(img, human.skeleton, min_limb_score_to_show)
    return img


def get_visualized_skeletons(img: np.ndarray, skeletons: List[SkeletonBase]) -> np.ndarray:
    for skeleton in skeletons:
        img = get_visualized_skeleton(img, skeleton)
    return img

''' 
    # FOR BLICKRICHTUNG ARROW
def get_visualized_skeleton(img: np.ndarray, skeleton: SkeletonBase, min_limb_score_to_show: float = 0.4):
    """
    Draws the skeletons joints and limbs in the image.
    :param img: The original image
    :param skeleton: The skeleton to be visualized
    :return: A copy of the image with the visualized skeleton
    """
    limb_line_width = 4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    img = np.asarray(img)
    for limb_num, limb in enumerate(skeleton.limbs):
        if not limb_should_be_displayed(limb, skeleton.limb_colors, min_limb_score_to_show):
            continue
        if limb_num == 24 : 
            img = __visualize_limb_arrow(img, limb, skeleton.limb_colors[limb_num], 5)
           #__visualize_limb(draw, limb, skeleton.limb_colors[limb_num], 6)
        else :
            __visualize_limb(draw, limb, skeleton.limb_colors[limb_num], limb_line_width)
    for joint_num, joint in enumerate(skeleton.joints):
        if joint.score >0.4:
            __visualize_joint(draw, joint, skeleton.joint_colors[joint_num], 3)
    # img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img



    # FOR NORMAL JOINTS AND LIMS
def get_visualized_skeleton(img: np.ndarray, skeleton: SkeletonBase, min_limb_score_to_show: float = 0.4):
    """
    Draws the skeletons joints and limbs in the image.
    :param img: The original image
    :param skeleton: The skeleton to be visualized
    :return: A copy of the image with the visualized skeleton
    """
    limb_line_width = 4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for limb_num, limb in enumerate(skeleton.limbs):
        if not limb_should_be_displayed(limb, skeleton.limb_colors, min_limb_score_to_show):
            continue
            __visualize_limb(draw, limb, skeleton.limb_colors[limb_num], 6)
        else :
            __visualize_limb(draw, limb, skeleton.limb_colors[limb_num], limb_line_width)
    for joint_num, joint in enumerate(skeleton.joints):
        if joint.score >0.4:
            __visualize_joint(draw, joint, skeleton.joint_colors[joint_num], 3)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
''' 


def get_visualized_skeleton(img: np.ndarray, skeleton: SkeletonBase, min_limb_score_to_show: float = 0.4):
    """
    Draws the skeletons joints and limbs in the image.
    :param img: The original image
    :param skeleton: The skeleton to be visualized
    :return: A copy of the image with the visualized skeleton
    """
    limb_line_width = 4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for limb_num, limb in enumerate(skeleton.limbs):
        if not limb_should_be_displayed(limb, skeleton.limb_colors, min_limb_score_to_show):
            continue
            __visualize_limb(draw, limb, skeleton.limb_colors[limb_num], 6)
        else :
            __visualize_limb(draw, limb, skeleton.limb_colors[limb_num], limb_line_width)
    for joint_num, joint in enumerate(skeleton.joints):
        if joint.score >0.4:
            __visualize_joint(draw, joint, skeleton.joint_colors[joint_num], 3)
            
            
    # img = np.asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.asarray(img)
    for limb_num, limb in enumerate(skeleton.limbs):
        if not limb_should_be_displayed(limb, skeleton.limb_colors, min_limb_score_to_show):
            continue
        if limb_num == 24 : 
            img = __visualize_limb_arrow(img, limb, skeleton.limb_colors[limb_num], 5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img
    
    
    


def visualize_limb(img: np.ndarray, limb: Limb2D, limb_color: Color, line_width: int = 4):
    """
    Visualizes the limb with the given color and line width.
    :param img: The original image
    :param limb: The limb to visualize
    :param limb_color: The color in which the limb should be displayed
    :param line_width: The width of the line visualizing the limb
    :return: The image with the visualized joints
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    __visualize_limb(draw, limb, limb_color, line_width)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def __visualize_limb(draw: ImageDraw, limb: Limb2D, limb_color: Color, line_width: int = 4):
    draw.line((int(limb.joint_from.x), int(limb.joint_from.y), int(limb.joint_to.x), int(limb.joint_to.y)),
              fill=limb_color.tuple_rgb,
              width=line_width)

              
def __visualize_limb_arrow(image, limb: Limb2D, limb_color: Color, line_width: int = 4):
    start_point = int(limb.joint_from.x), int(limb.joint_from.y)
    end_point = int(limb.joint_to.x), int(limb.joint_to.y)
    img = cv2.arrowedLine(image, start_point, end_point, color = limb_color.tuple_rgb, thickness = line_width)
    return img
    
    
def visualize_straying_joints(img: np.ndarray, straying_joint_dict: Dict[int, List[Joint2D]], joint_colors: List[Color]):
    """
    Visualizes joints which are not assigned to a skeleton. They will be displayed with gray color, background.
    :param img: The original image
    :param straying_joint_dict: dictionary with key: joint_num and value: List[Joint2D]
    :param joint_colors: The color list for each joint_num
    :return: The image with the visualized joints
    """
    for joint_num, straying_joints in straying_joint_dict.items():
        img = visualize_joints(img=img,
                               joints=straying_joints,
                               color=Colors.grey,
                               radius=10)
        img = visualize_joints(img=img,
                               joints=straying_joints,
                               color=joint_colors[joint_num],
                               radius=5)
    return img


def visualize_joint(img: np.ndarray, joint: Joint2D, color: Color, radius: int = 5):
    """
    Visualizes the given joint with the given color and radius.
    :param img: The original image
    :param joint: The joint to visualize
    :param color: The color in which the joints should be displayed
    :param radius: The radius of the joint circles
    :return: The image with the visualized joint
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    __visualize_joint(draw, joint, color, radius)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def __visualize_joint(draw: ImageDraw, joint: Joint2D, color: Color, radius: int = 5):
    draw.ellipse((int(joint.x) - radius, int(joint.y) - radius, int(joint.x) + radius, int(joint.y) + radius),
                 color.tuple_rgb)


def visualize_joints(img: np.ndarray, joints: List[Joint2D], color: Color, radius: int = 5):
    """
    Visualizes joints with the given color and radius.
    :param img: The original image
    :param joints: List of joints
    :param color: The color in which the joints should be displayed
    :param radius: The radius of the joint circles
    :return: The image with the visualized joints
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for joint in joints:
        __visualize_joint(draw, joint, color, radius=radius)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img




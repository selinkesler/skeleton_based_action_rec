import numpy as np
import cv2

from nobos_commons.data_structures.bounding_box import BoundingBox
from nobos_commons.data_structures.constants.color_palette import DETECTION_COLOR_PALETTE
from nobos_commons.data_structures.constants.detection_classes import COCO_CLASSES


def draw_bb(image: np.ndarray, bb: BoundingBox, title1: str = None, title2: str = None, title3: str = None, thickness: int = 2, text_size: int = 1,
            text_thickness: int = 1):
    """
    Draws the given bounding box in the image
    :param image: The image in which the bounding box should be drawn
    :param bb: The bounding box too be drawn
    :return: The image with the visualized bounding box
    """
    color = DETECTION_COLOR_PALETTE[COCO_CLASSES.index(bb.label)]
    top_left_tuple = (bb.top_left.x, bb.top_left.y)
    
    top_right_tuple = (bb.top_right.x, bb.top_right.y)
    
    bottom_right_tuple = (bb.bottom_right.x, bb.bottom_right.y)
    cv2.rectangle(image, top_left_tuple, bottom_right_tuple, color, thickness)
    text_box_size = cv2.getTextSize(bb.label, cv2.FONT_HERSHEY_PLAIN, text_size, text_thickness)[0]
    text_box_bottom_right = bb.top_left.x + text_box_size[0] + 95, bb.top_left.y + text_box_size[1] + 4
    cv2.rectangle(image, top_left_tuple, text_box_bottom_right, color, -1)
    
    text_box_2_top_left = bb.top_right.x - (text_box_size[0] + 3) , bb.top_right.y
    text_box_2_bottom_right = bb.top_right.x , bb.top_right.y + (text_box_size[1] + 4)
    cv2.rectangle(image, text_box_2_top_left, text_box_2_bottom_right, color, -1)
    
    print('bb : ',bb)
    
    box_title = bb.label
    if title1 is not None:
        box_title1 = title1
    cv2.putText(image, box_title1, (bb.top_left.x, bb.top_left.y + text_box_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, text_size, [225, 255, 255], text_thickness)
    if title3 is not None:
        box_title2 = title3
    cv2.putText(image, box_title2, (bb.top_right.x - (text_box_size[0] + 3),bb.top_right.y + (text_box_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, text_size, [225, 255, 255], text_thickness)
    
    return image
    
    
def draw_bb_stalker(image: np.ndarray, bb: BoundingBox, title1: str = None, title2: str = None, title3: str = None, thickness: int = 2, text_size: int = 1,
            text_thickness: int = 1):
    """
    Draws the given bounding box in the image
    :param image: The image in which the bounding box should be drawn
    :param bb: The bounding box too be drawn
    :return: The image with the visualized bounding box
    """
    color = DETECTION_COLOR_PALETTE[COCO_CLASSES.index("bicycle")]
    top_left_tuple = (bb.top_left.x, bb.top_left.y)
    
    top_right_tuple = (bb.top_right.x, bb.top_right.y)
    
    bottom_right_tuple = (bb.bottom_right.x, bb.bottom_right.y)
    cv2.rectangle(image, top_left_tuple, bottom_right_tuple, color, thickness)
    text_box_size = cv2.getTextSize(bb.label, cv2.FONT_HERSHEY_PLAIN, text_size, text_thickness)[0]
    text_box_bottom_right = bb.top_left.x + text_box_size[0] + 95, bb.top_left.y + text_box_size[1] + 4
    cv2.rectangle(image, top_left_tuple, text_box_bottom_right, color, -1)
    
    text_box_2_top_left = bb.top_right.x - (text_box_size[0] + 3) , bb.top_right.y
    text_box_2_bottom_right = bb.top_right.x , bb.top_right.y + (text_box_size[1] + 4)
    cv2.rectangle(image, text_box_2_top_left, text_box_2_bottom_right, color, -1)
    
    
    box_title = bb.label
    if title1 is not None:
        box_title1 = title1
    cv2.putText(image, box_title1, (bb.top_left.x, bb.top_left.y + text_box_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
                text_size, [225, 255, 255], text_thickness)
    if title3 is not None:
        box_title2 = title3
    cv2.putText(image, box_title2, (bb.top_right.x - (text_box_size[0] + 3),bb.top_right.y + (text_box_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN,
                text_size, [225, 255, 255], text_thickness)
    
    return image

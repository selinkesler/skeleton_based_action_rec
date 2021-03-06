from typing import List, Type

import numpy as np
import torch
from nobos_commons.data_structures.bounding_box import BoundingBox
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.skeletons.skeleton_base import SkeletonBase
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_torch_lib.configs.pose_estimation_2d_model_configs.pose_resnet_model_config import PoseResNetModelConfig
from nobos_torch_lib.models.pose_estimation_2d_models import pose_resnet
from nobos_torch_lib.models.pose_estimation_2d_models.pose_resnet import PoseResNet

# from ehpi_action_recognition.config import yolo_v3_config
# from ehpi_action_recognition.networks.detection_nets.detection_net_yolo_v3 import DetectionNetYoloV3, get_default_detector
from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_base import Pose2DNetBase
from ehpi_action_recognition.utils.pose_resnet_inference import get_human, get_human_pure, get_human_pure_w_bb


import gi
gi.require_version('Gtk', '2.0')
import os
import time
import json
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import trt_pose.coco
import trt_pose.models
import torch
from torch2trt import TRTModule
from PIL import Image


def get_image_content_from_humans(humans: List[Human]) -> ImageContent:
    bbs: List[BoundingBox] = []
    for human in humans:
        bbs.append(human.bounding_box)

    return ImageContent(humans=humans, objects=bbs)


class Pose2DNetResnet(Pose2DNetBase):
    __slots__ = ['model', 'skeleton_type', 'detector','topology','model_trt','human_pose']

    def __init__(self, skeleton_type: Type[SkeletonBase],human_pose):
        super().__init__(skeleton_type)
        self.human_pose = human_pose
        # self.detector: DetectionNetYoloV3 = get_default_detector(yolo_v3_config)

        # For new Pose Estimation
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        OPTIMIZED_MODEL = './resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        print(OPTIMIZED_MODEL)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        self.model = self.model_trt


    def get_humans_from_img(self, img: np.ndarray) -> List[Human]:
        bbs = self.detector.get_object_bounding_boxes(img)
        humans: List[Human] = []
        for bb in bbs:
            if bb.label is "person":
                human = get_human(self.model, self.skeleton_type, img, bb, self.human_pose, self.topology,self.model_trt)
                humans.append(human)

        return humans

    def get_humans_from_img_pure(self, img: np.ndarray,image_size) -> List[Human]:
        # bbs = self.detector.get_object_bounding_boxes(img)
        # humans: List[Human] = []
        humans = get_human_pure(self.model, self.skeleton_type, img, self.human_pose, self.topology,self.model_trt, image_size)
        return humans

    def get_humans_from_img_with_bb(self, img: np.ndarray, bbs: List[BoundingBox]) -> List[Human]:
        # bbs = self.detector.get_object_bounding_boxes(img)
        humans: List[Human] = []
        for bb in bbs:
            human,cond = get_human_pure_w_bb(self.model, self.skeleton_type, img, bb, self.human_pose, self.topology,self.model_trt)
            if cond == True :
                humans.append(human)
        return humans

    def redetect_humans(self, img: np.ndarray, humans_to_redetect, min_human_score: float = 0.0) -> List[Human]:
        humans: List[Human] = []
        for human_to_redetect in humans_to_redetect:
            human = get_human_pure(self.model, self.skeleton_type, img, self.human_pose, self.topology,self.model_trt)
            human.bounding_box = human_to_redetect.bounding_box
            human.uid = human_to_redetect.uid
            humans.append(human)
        return humans

    def get_humans_from_bbs(self, img: np.ndarray, bbs: List[BoundingBox], min_human_score: float = 0.55) -> List[Human]:
        humans: List[Human] = []
        # TODO: Run in parallel..
        for bb in bbs:
            if bb.label is "person":
                human = get_human(self.model, self.skeleton_type, img, bb, self.human_pose, self.topology,self.model_trt)
                if human.score < min_human_score:
                    continue
                human.bounding_box = bb
                humans.append(human)

        return humans


def get_default_network(pose_resnet_config: PoseResNetModelConfig) -> Pose2DNetResnet:
    model = pose_resnet.get_pose_net(pose_resnet_config)

    # logger.info('=> loading model from {}'.format(cfg.pose_estimator.model_state_file))
    model.load_state_dict(torch.load(pose_resnet_config.model_state_file))

    model = model.cuda()
    model.eval()
    return Pose2DNetResnet(model, SkeletonStickman)

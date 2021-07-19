# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import math
from typing import Type

import cv2
import numpy as np
import torch
from nobos_commons.data_structures.bounding_box import BoundingBox
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.skeletons.joint_visibility import JointVisibility
from nobos_commons.data_structures.skeletons.skeleton_base import SkeletonBase
from nobos_commons.data_structures.skeletons.skeleton_coco import SkeletonCoco
from nobos_commons.tools.skeleton_converters.skeleton_converter_factory import SkeletonConverterFactory
from nobos_torch_lib.models.pose_estimation_2d_models.pose_resnet import PoseResNet
from nobos_torch_lib.utils.yolo_helper import bb_to_center_scale
from torchvision.transforms.functional import to_tensor, normalize
from nobos_commons.data_structures.dimension import Coord2D

from ehpi_action_recognition.config import pose_resnet_config
from ehpi_action_recognition.utils.pose_resnet_transforms import get_affine_transform, transform_preds
from nobos_commons.utils.bounding_box_helper import get_human_bounding_box_from_joints

skeleton_converter_factory = SkeletonConverterFactory()

# For New Pose Estimation
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


WIDTH = 224
HEIGHT = 224
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


def get_human_pose_result(coco_skeleton: SkeletonCoco, skeleton_type: Type[SkeletonBase]) -> Human:
    skeleton_converter = skeleton_converter_factory.get_skeleton_converter(SkeletonCoco, skeleton_type)
    skeleton = skeleton_converter.get_converted_skeleton(coco_skeleton)
    human = Human(skeleton=skeleton)
    return human


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return  image[None, ...]

def process_img(image,human_pose, topology, model_trt):

    # NEW POSE ESTIMATION from pose_trt
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    image = cv2.resize(image, (WIDTH, HEIGHT))

    data = preprocess(image)
    start = time.time()
    start_model = time.time()
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

    peaks_lis = peaks[0].tolist()
    n_people = int(counts)
    return peaks_lis,peaks, n_people, objects,counts

def get_human(model: PoseResNet, skeleton_type: Type[SkeletonBase], image: np.ndarray, bb: BoundingBox, human_pose, topology, model_trt) -> Human:
    if bb.label != "person":
        return None
    with torch.no_grad():
        skeleton_coco: SkeletonCoco = SkeletonCoco


        # OLD VERRSION
        #####################
        center, scale = bb_to_center_scale(bb)
        r = 0
        image_size = [pose_resnet_config.input_height, pose_resnet_config.input_width]
        trans = get_affine_transform(center, scale, r, image_size)
        net_input = cv2.warpAffine(
            image,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
      
        net_input = to_tensor(net_input)
        net_input = normalize(net_input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        net_input = net_input.unsqueeze(0)
        #####################

        peaks_lis,n_people = process_img(image, human_pose, topology, model_trt)

        for joint_num in range (0, len(peaks_lis)-1):
            skeleton_coco.joints[joint_num].y = peaks_lis[joint_num][0][0] * 480
            skeleton_coco.joints[joint_num].x = peaks_lis[joint_num][0][1] * 640
            skeleton_coco.joints[joint_num].score = 0.8
            if skeleton_coco.joints[joint_num].x == 0 and skeleton_coco.joints[joint_num].y == 0:
                skeleton_coco.joints[joint_num].visibility = JointVisibility.ABSENT
            else:
                skeleton_coco.joints[joint_num].visibility = JointVisibility.VISIBLE    

        human = get_human_pose_result(skeleton_coco, skeleton_type)
        human.bounding_box = bb
        return human


def get_human_pure(model: PoseResNet, skeleton_type: Type[SkeletonBase], image: np.ndarray, human_pose, topology, model_trt, image_size) -> Human:
    with torch.no_grad():

        width = image_size.width
        height = image_size.height

        skeleton_coco: SkeletonCoco = SkeletonCoco
        peaks_lis,peaks, n_people, objects,counts = process_img(image, human_pose, topology, model_trt)
        humans: List[Human] = []
        print('n_people : ', n_people)

        draw_objects = DrawObjects(topology)

        for i in range (n_people):
            skeleton_coco: SkeletonCoco = SkeletonCoco
            obj = objects[0][i]
            C = obj.shape[0]
            visible = 0
            for j in range(C-1):
                k = int(obj[j])
                if k < 100000:
                    peak = peaks[0][j][k]
                    skeleton_coco.joints[j].x = round(float(peak[1]) * width)
                    skeleton_coco.joints[j].y = round(float(peak[0]) * height)
                    skeleton_coco.joints[j].score = 0.8
                    if skeleton_coco.joints[j].x == 0 and skeleton_coco.joints[j].y == 0:
                        skeleton_coco.joints[j].visibility = JointVisibility.ABSENT
                    else:
                        skeleton_coco.joints[j].visibility = JointVisibility.VISIBLE   
                        visible += 1 

            print('visible : ',visible)
            if visible >= 8 :
                human = get_human_pose_result(skeleton_coco, skeleton_type)

                bb = get_human_bounding_box_from_joints(skeleton_coco.joints, width, height)

                human.bounding_box = bb
                print('human being appended')
                humans.append(human)

    return humans

def get_human_pure_w_bb(model: PoseResNet, skeleton_type: Type[SkeletonBase], image: np.ndarray, bb: BoundingBox, human_pose, topology, model_trt):
    ################## BB Resize scale stuff are not working ##################
    with torch.no_grad():
        skeleton_coco: SkeletonCoco = SkeletonCoco
        # OLD VERRSION
        #####################
        center, scale = bb_to_center_scale(bb)
        r = 0
        image_size = [224, 224]
        trans = get_affine_transform(center, scale, r, image_size)
        net_input = cv2.warpAffine(
            image,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
            
        peaks_lis,n_people = process_img(net_input, human_pose, topology, model_trt)
        # humans: List[Human] = []
        print('n_people : ', n_people)

        visible = 0
        for joint_num in range (0, len(peaks_lis)-1):
            skeleton_coco.joints[joint_num].y = int(peaks_lis[joint_num][0][0] * bb.height) # *(256/bb.height)
            skeleton_coco.joints[joint_num].x = int(peaks_lis[joint_num][0][1] * bb.width) # *(192/bb.width)

            skeleton_coco.joints[joint_num].score = 0.8
            if skeleton_coco.joints[joint_num].x == 0 and skeleton_coco.joints[joint_num].y == 0:
                skeleton_coco.joints[joint_num].visibility = JointVisibility.ABSENT
            else:
                skeleton_coco.joints[joint_num].visibility = JointVisibility.VISIBLE   
                visible += 1 

        print('visible : ',visible)
        if visible >= 8 :
            human = get_human_pose_result(skeleton_coco, skeleton_type)

            # bb = get_human_bounding_box_from_joints(skeleton_coco.joints, 640, 480)
            human.bounding_box = bb
            print('human being appended')
            cond = True

            return human, cond
        else :
            cond = False
            human = None
            return human, cond

    '''
        # OLD VERRSION
        center, scale = bb_to_center_scale(bb)
        r = 0
        image_size = [pose_resnet_config.input_height, pose_resnet_config.input_width]
        trans = get_affine_transform(center, scale, r, image_size)
        net_input = cv2.warpAffine(
            image,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        net_input = to_tensor(net_input)
        net_input = normalize(net_input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        net_input = net_input.unsqueeze(0)

        output = model(net_input.cuda())
        # print('output')

        # print(output)
        print(output.shape[2])
        heatmaps = output.clone().cpu().numpy()
        # heatmaps = output.clone()
        preds, joint_scores = get_final_preds(heatmaps, [center], [scale])
        skeleton_coco: SkeletonCoco = SkeletonCoco
        preds = preds[0].tolist()
        assert len(joint_scores) <= 1, "Joints for more than one human .."

        for joint_num, joint_score in enumerate(joint_scores[0]):
            assert len(joint_score) <= 1, "More than one joint score!!"
            skeleton_coco.joints[joint_num].x = preds[joint_num][0]
            skeleton_coco.joints[joint_num].y = preds[joint_num][1]
            skeleton_coco.joints[joint_num].score = joint_score[0].item()
            skeleton_coco.joints[joint_num].visibility = JointVisibility.VISIBLE

        human = get_human_pose_result(skeleton_coco, skeleton_type)
        human.bounding_box = bb
        return human
    '''


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    
    preds *= pred_mask
    
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, joint_scores = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if pose_resnet_config.post_process:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, joint_scores

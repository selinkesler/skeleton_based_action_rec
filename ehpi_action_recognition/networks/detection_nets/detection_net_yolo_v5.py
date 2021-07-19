import torch
import cv2
from typing import List
import argparse
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from nobos_commons.data_structures.bounding_box import BoundingBox
from nobos_commons.data_structures.constants.detection_classes import COCO_CLASSES
from nobos_commons.data_structures.dimension import Coord2D
from nobos_torch_lib.configs.detection_model_configs.yolo_v3_config import YoloV3Config
from nobos_torch_lib.models.detection_models.yolo_v3 import Darknet
from nobos_torch_lib.utils.yolo_helper import write_results
from torch.autograd import Variable

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.dimension import ImageSize
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from ehpi_action_recognition.config import yolo_v3_config

class DetectionNetYoloV3(object):
    __slots__ = ['model', 'num_classes','webcam','save_img','weights','view_img', 'save_txt','imgsz','project','name','exist_ok','augment', \
        'conf_thres','iou_thres','classes','agnostic_nms','save_conf','half','modelc','names','colors','classify']

    '''
    def __init__(self, model: Darknet):
        self.model = model
        self.num_classes = len(COCO_CLASSES)
        self.model.net_info["width"] = self.model.net_info["height"] = yolo_v3_config.resolution

        assert yolo_v3_config.resolution % 32 == 0
        assert yolo_v3_config.resolution > 32

        if yolo_v3_config.use_gpu:
            model.cuda()

        model.eval()
    '''


    def __init__(self):

        self.webcam = False
        self.save_img = False
        self.num_classes = len(COCO_CLASSES)

        print('nbr')
        # source = '0'
        
        self.weights = 'yolov5s.pt'
        self.view_img = 'store_true'
        self.save_txt = 'store_true'
        self.imgsz = 640
        
        
        self.project = 'runs/detect'
        self.name = 'exp' 
        
        
        self.exist_ok = 'store_true'
        self.augment = 'store_true'
        self.conf_thres = 0.25
        self.iou_thres = 0.45    
        
        
        # number = 2
        # print(type(number))
        self.classes = 0
        self.agnostic_nms = 'store_true'
        self.save_conf = 'store_true'
    
        # Initialize
        set_logging()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = select_device(opt.device) ## original
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        self.model = attempt_load(self.weights, map_location=device)  # load FP32 model
        # imgsz = ImageSize(width=1280, height=720)
        imgsz = 640
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        
        if self.half:
            self.model.half()  # to FP16
    
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
    
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = self.model(img.half() if self.half else img) if device.type != 'cpu' else None  # run once

        self.model.eval()

    def get_object_bounding_boxes(self, image: np.ndarray) -> List[BoundingBox]:
        network_input = self._get_network_input(image)
        if yolo_v3_config.use_gpu:
            network_input = network_input.cuda()
        output = self.model(Variable(network_input))
        output = write_results(output, yolo_v3_config.confidence, self.num_classes, nms=True,
                               nms_thresh=yolo_v3_config.nms_thresh)

                            

        if output is 0:
            return []
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(yolo_v3_config.resolution)) / yolo_v3_config.resolution

        #            im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= image.shape[1]
        output[:, [2, 4]] *= image.shape[0]
        bbs: List[BoundingBox] = []
        for x in output:
            top_left = tuple(x[1:3].int())
            bottom_right = tuple(x[3:5].int())
            top_left = Coord2D(x=top_left[0].item(), y=top_left[1].item())
            bottom_right = Coord2D(x=bottom_right[0].item(), y=bottom_right[1].item())
            class_id = int(x[-1].cpu())
            class_label = "{0}".format(COCO_CLASSES[class_id])
            bb = BoundingBox(top_left, bottom_right, label=class_label)
            if bb.width > 0 and bb.height > 0:
                bbs.append(bb)
        return bbs

        
        
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

        
        
    def detect(self, source_img: np.ndarray ) -> List[BoundingBox]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        img_size = 640

        # img0 = cv2.imread(path)  # BGR
        img0 = source_img

        # Padded resize
        img = self.letterbox(img0, new_shape=img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # return img, img0

        img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # pred = model(img, augment=opt.augment)[0] ## origninal
        pred = self.model(img, augment=self.augment)[0]

        ####################### ALTERNATIVE 1 ########################
        # not from yolov5 directly but from the yolov3 helper
        
        output = write_results(pred, yolo_v3_config.confidence, self.num_classes, nms=True,
                               nms_thresh=yolo_v3_config.nms_thresh)
        image = source_img  

        if output is 0:
            return []
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(yolo_v3_config.resolution)) / yolo_v3_config.resolution

        #            im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= image.shape[1]
        output[:, [2, 4]] *= image.shape[0]
        bbs: List[BoundingBox] = []
        for x in output:
            top_left = tuple(x[1:3].int())
            bottom_right = tuple(x[3:5].int())
            top_left = Coord2D(x=top_left[0].item(), y=top_left[1].item())
            bottom_right = Coord2D(x=bottom_right[0].item(), y=bottom_right[1].item())
            class_id = int(x[-1].cpu())
            class_label = "{0}".format(COCO_CLASSES[class_id])
            bb = BoundingBox(top_left, bottom_right, label=class_label)
            if bb.width > 0 and bb.height > 0:
                bbs.append(bb)
        return bbs




        ####################### ALTERNATIVE 2 ########################
        # with original code from yolov5 updated for our case
        '''
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        for i, det in enumerate(pred):
            #im0 = img0
            bbs: List[BoundingBox] = []
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size                
                # print(det[:, -1])
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() ##
                
                for count,action_id in enumerate(det[:, -1]) : 
                    class_name = self.names[int(action_id)]
                    # print(str(count) + class_name)
                    # print(det[:, :4][count][0])
                    
                    if class_name == "person" : 
                    
                        top_left_x = int(det[:, :4][count][0])
                        top_left_y = int(det[:, :4][count][1])
                        bottom_right_x = int(det[:, :4][count][2])
                        bottom_right_y = int(det[:, :4][count][3])
                        
                        # top_left = (top_left_x, top_left_y)
                        top_left = Coord2D(top_left_x, top_left_y)
                        bottom_right = Coord2D(bottom_right_x, bottom_right_y) 
                        class_label = class_name
                        bb = BoundingBox(top_left, bottom_right, label=class_label)
                        if bb.width > 0 and bb.height > 0:
                            bbs.append(bb)
            '''

        # return bbs


    @staticmethod
    def _get_network_input(image: np.ndarray):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """

        net_input_image = cv2.resize(image, (yolo_v3_config.resolution, yolo_v3_config.resolution))
        net_input_image = net_input_image[:, :, ::-1].transpose((2, 0, 1)).copy()
        net_input_image = torch.from_numpy(net_input_image).float().div(255.0).unsqueeze(0)
        return net_input_image


def get_default_detector(cfg: YoloV3Config) -> DetectionNetYoloV3:
    yolo_model = Darknet(cfg)
    yolo_model.load_weights(cfg.model_state_file)
    return DetectionNetYoloV3(model=yolo_model)


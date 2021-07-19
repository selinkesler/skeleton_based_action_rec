#!/usr/bin/env python3
import gi
gi.require_version('Gtk', '2.0')
import os
import time
import json
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import trt_pose.coco
import trt_pose.models
import torch
from torch2trt import TRTModule
from PIL import Image
import cv2
# import matplotlib
# matplotlib.use('Agg') # due to gtk version problems as a solition

WIDTH = 224
HEIGHT = 224
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
print(OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')

    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def process_img(image):
    ori_image = image.copy()
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data = preprocess(image)
    start = time.time()
    start_model = time.time()
    cmap, paf = model_trt(data)
    print("FPS model: ", 1.0/(time.time() - start_model))
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    print('cmap : ',cmap)
    print('paf : ', paf)
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    print("FPS: ", 1.0/(time.time() - start))
    # print('counts : ', counts)
    # print('objects : ', objects)


    preds = objects[0].tolist()
    # peaks_lis = peaks[0].tolist()
    # print('peaks_lis : ', peaks_lis)
    draw_objects(ori_image, counts, objects, peaks)
    return ori_image

def predict_image(path = '1.jpg'):
    image = cv2.imread(path)
    img = process_img(image)
    cv2.imshow("as", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def predict_video(path_video):
    print(path_video)
    if os.path.exists(path_video):
        print("exist path video")
        vid = cv2.VideoCapture(path_video)

    while(True):
        ret, frame = vid.read() 
        if not ret:
            # print("no frame")
            break

        frame = process_img(frame)
        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow("as", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    vid.release()
    cv2.destroyAllWindows() 

predict_video('video.MOV')
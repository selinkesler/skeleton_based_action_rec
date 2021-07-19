
from collections import deque
from operator import itemgetter
from typing import Dict, List, Tuple
#from ehpi_action_recognition.config import pose_resnet_config, pose_visualization_config, ehpi_model_state_file
from config import pose_resnet_config, pose_visualization_config, ehpi_model_state_file


import cv2
import numpy as np
import torch.utils.data.distributed

from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.image_content_buffer import ImageContentBuffer
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import get_joints_jhmdb
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import FeatureVecProducerEhpi
from nobos_commons.input_providers.camera.webcam_provider import WebcamProvider
from nobos_commons.tools.fps_tracker import FPSTracker
from nobos_commons.tools.log_handler import logger
from nobos_commons.tools.pose_tracker import PoseTracker
from nobos_commons.visualization.detection_visualizer import draw_bb
from nobos_commons.visualization.pose2d_visualizer import get_human_pose_image
from nobos_torch_lib.models.detection_models.shufflenet_v2 import ShuffleNetV2
from nobos_torch_lib.models.pose_estimation_2d_models import pose_resnet
from scipy.special import softmax



#from ehpi_action_recognition.configurator import setup_application
#from ehpi_action_recognition.networks.action_recognition_nets.action_rec_net_ehpi import ActionRecNetEhpi
#from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet

from configurator import setup_application
from networks.action_recognition_nets.action_rec_net_ehpi import ActionRecNetEhpi
#from networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet

action_save: Dict[str, List[List[float]]] = {}

print("hi")
import os
import torch.backends.cudnn

from nobos_commons.data_structures.configs.cache_config import CacheConfig
from nobos_commons.data_structures.configs.pose_visualization_config import PoseVisualizationConfig
from nobos_torch_lib.configs.detection_model_configs.yolo_v3_config import YoloV3Config
from nobos_torch_lib.configs.pose_estimation_2d_model_configs.pose_resnet_model_config import PoseResNetModelConfig

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir, "data")
models_dir = os.path.join(data_dir, "models")

# Cache Config
cache_config = CacheConfig(cache_dir=os.path.join(data_dir, "cache"),
                           func_names_to_reload=[],
                           reload_all=False)

# Torch Backend Settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

# Models Configs
# Pose Recognition
pose_resnet_config = PoseResNetModelConfig()
pose_resnet_config.model_state_file = os.path.join(data_dir, "models", "pose_resnet_50_256x192.pth.tar")

# Object Detection
yolo_v3_config = YoloV3Config()
yolo_v3_config.network_config_file = os.path.join(curr_dir, "configs", "yolo_v3.cfg")
yolo_v3_config.model_state_file = os.path.join(models_dir, "yolov3.weights")
yolo_v3_config.resolution = 160

# Action Recognition

# ShuffleNet , 7 Class
#ehpi_model_state_file = os.path.join(models_dir, "ehpi_model_NTU_0.01_128_cp0101_checkpoint0060.pth")
ehpi_model_state_file = os.path.join(models_dir, "7_class_shuffle.pth")


# ShuffleNet , 5 Class
#ehpi_model_state_file = os.path.join(models_dir, "ehpi_model_NTU_0.01_128_cp0101_checkpoint0060.pth")
ehpi_model_state_file_5 = os.path.join(models_dir, "5_class_shuffle_SGD_16_add____90.pth")
ehpi_model_state_file_3 = os.path.join(models_dir, "3_class_shuffle_SGD_16_add____60.pth")
ehpi_model_custom_4_SGD = os.path.join(models_dir, "CUSTOM_4_TRAIN_ONLY_SGD/CUSTOM_4_0.01_64_cp0151_checkpoint0150.pth")
ehpi_model_custom_4_ADAM = os.path.join(models_dir, "CUSTOM_4_TRAIN_ONLY_ADAM/CUSTOM_4_ADAM0.01_64_cp0151_checkpoint0150.pth")

# LSTM , 7 Class
# ehpi_model_state_file = os.path.join(models_dir, "ehpi_model_NTU_0.01_128_cp0151_checkpoint0150.pth")

# original 3 classes
# ehpi_model_state_file = os.path.join(models_dir, "ehpi_v1.pth")
ehpi_dataset_path = os.path.join(data_dir, "datasets")

# Visualization
pose_visualization_config = PoseVisualizationConfig()

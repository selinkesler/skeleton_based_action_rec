import os
import time

import cv2
import numpy as np
from queue import Queue
from threading import Thread

from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.input_providers.input_provider_base import InputProviderBase
from nobos_commons.utils.file_helper import get_img_paths_from_folder

# IMPORTING ROS MODULE
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# to try with saved video file in computer


from nobos_commons.utils.file_helper import get_video_paths_from_folder




 
class VideoDirProvider(InputProviderBase):

    
    
    # WebCamProvider'dan calinmistir :) 
    def __init__(self,
                 camera_number: str,
                 image_size: ImageSize = ImageSize(width=640, height=480),
                 fps: int = 30):
        """
        Provides frames captured from a webcam. Uses OpenCV internally.
        :param camera_number: The cameras id
        :param image_size: The image size which should be used, may be limited by camera parameters
        :param fps: The fps on which the frames should be grabbed, may be limited by camera parameters
        """
        self.cap = cv2.VideoCapture(camera_number)
        self.image_size = image_size
        

    def get_data(self) -> np.ndarray:
        assert self.cap.isOpened(), 'Cannot capture source'
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                print(frame.shape)
                yield frame

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    return None

            else:
                print("agy")
                return None

        self.cap.release()

    def stop(self):
        self.cap.release()

        
        
        
        
           
class StereoDirProvider(InputProviderBase):
    
    def __init__(self,
                 image_size: ImageSize = ImageSize(width=1600, height=1300),
                 fps: int = 30):
        self.x1 = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(0.5)
        
    def showImage(self,img):
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def image_callback(self, msg):
        #rospy.loginfo("x1: {}".format(self.x1))
        bridge = CvBridge()

        try:
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            self.x1 = cv2_img
            # rospy.loginfo("x1: {}".format(self.x1))
            
            # to display the video with opencv : 
            self.showImage(cv2_img)
            print(cv2_img.shape)
        
        except CvBridgeError as e:
            print(e)
        else:
            # print("elsedeyim")
            cv2.imwrite('camera_image.jpeg',cv2_img)        
        
    # rospy.init_node('image_listener')    
    
    def get_data(self):
    
        rospy.loginfo("In attesa")
    
        #rospy.init_node('image_listener')
        image_topic="/taraxl/left/image_rect"
 
        while not rospy.is_shutdown():
            rospy.Subscriber(image_topic,Image,self.image_callback)
            # print("lütfen x1:")
            # print(self.x1)
            frame = self.x1
            yield frame
            # rospy.loginfo("x1: {}".format(self.x1))
            self.loop_rate.sleep()
            # rospy.spin()

    def stop(self):
        self.cap.release()       

        
class VideoDirProviderAsync(InputProviderBase):
    def __init__(self,
                 camera_number: str,
                 image_size: ImageSize = ImageSize(width=1280, height=720),
                 fps: int = 30,
                 queue_size = 128):
        """
        Provides frames captured from a webcam. Uses OpenCV internally.
        :param camera_number: The cameras id
        :param image_size: The image size which should be used, may be limited by camera parameters
        :param fps: The fps on which the frames should be grabbed, may be limited by camera parameters
        """
        self.cap = cv2.VideoCapture(camera_number)
        self.stopped = True
        self.queue = Queue(maxsize=queue_size)
        self.image_size = image_size

    def start(self):
        self.stopped = False
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        assert self.cap.isOpened(), 'Cannot capture source'

        while self.cap.isOpened():
            if self.stopped:
                return None
            if not self.queue.full():
                ret, frame = self.cap.read()
                if ret:
                    self.queue.put(frame)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        self.stop()
                        return None

                else:
                    self.stop()
                    return None

        self.cap.release()

    def get_data(self) -> np.ndarray:
        assert self.cap.isOpened(), 'Cannot capture source'
        if self.stopped:
            self.start()
        while True:
            if self.stopped:
                return None
            if self.queue.qsize() > 0:
                yield self.queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()




class ImgDirProvider(InputProviderBase):
    def __init__(self,
                 img_dir: str,
                 fps: int = None,
                 loop: bool = False,
                 image_size: ImageSize = None,
                 print_current_path: bool = False):
        """
        Provides images from a given image directory.
        :param img_dir: The directory which contains the images
        :param fps: Forced fps for the image delivery. May be limited by disk io...
        """
        self.img_dir = img_dir
        assert os.path.exists(img_dir), 'Image directory not found!'
        self.img_paths = sorted(get_img_paths_from_folder(img_dir))
        self.fps = fps
        self.loop = loop
        self.image_size = image_size
        self.print_current_path = print_current_path
        self.stopped = False

    def get_data(self) -> np.ndarray:
        assert len(self.img_paths) > 0, 'No images found'
        while True:
            if self.stopped:
                return None
            for img_path in self.img_paths:
                if self.fps is not None:
                    start = time.time()
                if self.print_current_path:
                    print(img_path)
                img = cv2.imread(img_path)
                if self.image_size is not None:
                    img = cv2.resize(img, (self.image_size.width, self.image_size.height))
                yield img

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    return None

                if self.fps is not None:
                    time.sleep(max(1. / self.fps - (time.time() - start), 0))
            if not self.loop:
                break

    def stop(self):
        self.stopped = True

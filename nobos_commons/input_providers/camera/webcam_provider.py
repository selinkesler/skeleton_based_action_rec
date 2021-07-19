from queue import Queue
from threading import Thread

import cv2
import numpy as np

from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.input_providers.input_provider_base import InputProviderBase
import pyrealsense2 as rs

class WebcamProvider(InputProviderBase):
    def __init__(self,
                 camera_number: int = 0,
                 image_size: ImageSize = ImageSize(width=1280, height=720),
                 fps: int = 60,
                 mirror: bool = False):
        """
        Provides frames captured from a webcam. Uses OpenCV internally.
        :param camera_number: The cameras id
        :param image_size: The image size which should be used, may be limited by camera parameters
        :param fps: The fps on which the frames should be grabbed, may be limited by camera parameters
        """
        
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)


        #self.cap = cv2.VideoCapture(camera_number)
        #self.mirror = mirror
        #self.image_size = image_size
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height)
        #self.cap.set(cv2.CAP_PROP_FPS, fps)


        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        # self.cap.set(cv2.CAP_PROP_FOCUS, 0.0)

    def get_data(self) -> np.asanyarray:
        # assert self.cap.isOpened(), 'Cannot capture source'

        # while self.cap.isOpened():
            # ret, frame = self.cap.read()
            # if self.mirror:
                # frame = cv2.flip(frame, 1)
            # if ret:
                # yield frame

                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                    # return None

            # else:
                # return None

        # self.cap.release()
         while True:

            # Wait for a color frame
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: #or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            yield color_image
            
            #cv2.imshow('RealSense', color_image)
            #key = cv2.waitKey(1)
            
            #if key == 27:
                #cv2.destroyAllWindows()
                #return None
                    
                    
    def stop(self):
        self.pipeline.stop()

class WebcamProviderAsync(InputProviderBase):
    def __init__(self,
                 camera_number: int = 0,
                 image_size: ImageSize = ImageSize(width=1280, height=720),
                 fps: int = 60,
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

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

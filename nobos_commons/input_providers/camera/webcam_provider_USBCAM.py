
from jetcam.usb_camera import USBCamera
import gi
gi.require_version('Gtk', '2.0')
from queue import Queue
from threading import Thread
import cv2
import numpy as np

from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.input_providers.input_provider_base import InputProviderBase
import pyrealsense2 as rs

class WebcamProvider_USBCAM(InputProviderBase):
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
        self.image_size = image_size
        self.camera = USBCamera(capture_device=camera_number, width=self.image_size.width, height = self.image_size.height)

    def get_data(self) -> np.asanyarray:

         while True:
            self.camera.read()
            frame = self.camera.value
            yield frame
            
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

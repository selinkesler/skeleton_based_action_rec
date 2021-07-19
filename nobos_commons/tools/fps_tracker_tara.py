import time


class FPSTracker(object):
    def __init__(self):
        # self.average_over_seconds = average_over_seconds
        # self.frame_counter = 0
        # self.reference_start_time = None
        self.current_fps = -1

    def get_fps(self) -> float :    
        self.current_fps = 1
        return self.current_fps

    def print_fps(self):
        # fps = self.get_fps()
        print("FPS of the video")

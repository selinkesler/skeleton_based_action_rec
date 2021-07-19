#!/usr/bin/env python3.6

import rospy
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.clock import Clock
from kivy.clock import mainthread
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.widget import Widget 
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config

from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import UInt16
from std_msgs.msg import Float64

import subprocess
import threading
import multiprocessing
import time
import os
import signal
#import psutil

#########################################################################################################################
#												working with multiple screens
#########################################################################################################################

Window.size = (800, 480)
# Window.fullscreen = 'auto'


from kivy.uix.screenmanager import (ScreenManager, Screen, NoTransition, 
SlideTransition, CardTransition, SwapTransition, 
FadeTransition, WipeTransition, FallOutTransition, RiseInTransition)
Builder.load_file("my.kv")


# Create a class for all screens in which you can include 
# helpful methods specific to that screen 
class ScreenZero(Screen): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
    pass

class ScreenOne(Screen): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
    pass

   
class ScreenTwo(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
    pass
  
class ScreenThree(Screen): 
    pass
   
  

class TutorialApp(MDApp): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = ScreenManager(transition = NoTransition()) 
        
        # Add the screens to the manager and then supply a name 
        # that is used to switch screens 
        self.screen_manager.add_widget(ScreenZero(name ="screen_zero")) 
        self.screen_manager.add_widget(ScreenOne(name ="screen_one")) 
        self.screen_manager.add_widget(ScreenTwo(name ="screen_two")) 
        self.screen_manager.add_widget(ScreenThree(name ="screen_three"))         

        self.sec_0 = self.screen_manager.get_screen("screen_zero")  
        self.sec_1 = self.screen_manager.get_screen("screen_one")   
        self.sec_2 = self.screen_manager.get_screen("screen_two")   
        self.sec_3 = self.screen_manager.get_screen("screen_three")
        
    def build(self): 

        self.sec_1.ids.battery_im.source = "battery/zero.png"
        self.sec_1.ids.network_im.source = "network_quality/loading.jpg"
                
        self.msg_with_action_funtion = False
        self.msg_without_action_funtion = False
        self.msg_with_new_action_funtion = False

        self.autonomus = False
        self.manuel = False
        self.action_rec = False
        self.floor = False

        on_top = "wmctrl -r Tutorial -b add,above"
        self.top = subprocess.Popen(['xterm', '-e', on_top])

        self.event_with_action = Clock.schedule_interval(self.battery, 1)
                  
        return self.screen_manager

    def autonomus_mode_function(self, *args):

        cmd_aut_1 = "roslaunch human_stalker follow_person.launch"
        cmd_aut_2 = "roslaunch epic_car_description create_model.launch" 
        cmd_aut_3 = "roslaunch realsense2_camera rs_camera.launch"  ##### not neccessary for local processing, only for image transfer to Trainings PC
        cmd_aut_4 = "rosrun human_stalker image_resize.py"  # Launch Camera Image Resizer & Compressor ##### not neccessary for local processing, only for image transfer to Trainings PC
        cmd_aut_5 = "rosrun image_transport republish raw in:=image_resized out:=/image_compressed" ##### not neccessary for local processing, only for image transfer to Trainings PC
        cmd_aut_6 = "roslaunch epic_drive_functions cartographer.launch" # Launch Cartographer
        cmd_aut_7 = "rosrun obstacle_avoidance info_uss.py" # Launch Obstace Avoidance (Utraschall Sensoren Arduin Communication)
        

        self.aut_1 = subprocess.Popen(['xterm', '-e', cmd_aut_1])
        self.aut_2 = subprocess.Popen(['xterm', '-e', cmd_aut_2])
        # self.aut_3 = subprocess.Popen(['xterm', '-e', cmd_aut_3])
        # self.aut_4 = subprocess.Popen(['xterm', '-e', cmd_aut_4])
        # self.aut_5 = subprocess.Popen(['xterm', '-e', cmd_aut_5])
        self.aut_6 = subprocess.Popen(['xterm', '-e', cmd_aut_6])
        # self.aut_7 = subprocess.Popen(['xterm', '-e', cmd_aut_7])

        self.autonomus = True

        print("autonomus_mode_function button_pressed")

    def manuel_mode_function(self, *args):

        cmd_man_1 = "roslaunch epic_drive_functions joystick_teleop.launch"
        cmd_man_2 = "roslaunch taraxl_ros_package taraxl.launch" # Launch taraxl_ros_package
        cmd_man_3 = "rosrun human_stalker image_resize.py"  # Launch Camera Image Resizer & Compressor ##### not neccessary for local processing, only for image transfer to Trainings PC
        cmd_man_4 = "rosrun image_transport republish raw in:=image_resized out:=/image_compressed" ##### not neccessary for local processing, only for image transfer to Trainings PC
        cmd_man_5 = "roslaunch epic_drive_functions cartographer.launch" # Launch Cartographer

        self.man_1 = subprocess.Popen(['xterm', '-e', cmd_man_1])
        self.man_2 = subprocess.Popen(['xterm', '-e', cmd_man_2])
        # self.man_3 = subprocess.Popen(['xterm', '-e', cmd_man_3])
        # self.man_4 = subprocess.Popen(['xterm', '-e', cmd_man_4])
        self.man_5 = subprocess.Popen(['xterm', '-e', cmd_man_5])

        self.manuel = True
        print("manuel_mode_function button_pressed")

    def with_action_function(self, *args):
        print("with_action_function button_pressed")
        
        self.sec_1.ids.floor_fuction_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        self.sec_1.ids.with_new_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        self.sec_1.ids.stop_button.pos_hint = {"center_x": 0.5, "center_y": 0.3}
        
        self.msg_with_action_funtion = True
        self.msg_without_action_funtion = False
        self.msg_with_new_action_funtion = False
        
        with_action_pub.publish(self.msg_with_action_funtion)
        with_action_pub_2.publish(self.msg_with_action_funtion)
        without_action_pub.publish(self.msg_without_action_funtion)
        new_action_pub.publish(self.msg_with_new_action_funtion)

        cmd = "rosrun gui run_ehpi_fast.py"
        # program = subprocess.Popen(cmd, start_new_session=True)
        # program = subprocess.Popen(['gnome-terminal',cmd], stdout=subprocess.PIPE, shell=True)
        # self.p = subprocess.Popen(['gnome-terminal','rosrun gui run_ehpi_class_7_ROS.py'])
        self.action_rec = True
        self.p = subprocess.Popen(['xterm', '-e', cmd])

        self.sec_1.ids.with_action_function_button.disabled = True

        self.event_with_action = Clock.schedule_interval(self.listener_with_action, 1)
               
    def floor_fuction(self, *args):
        print("floor_fuction button_pressed")

        # to get rid of other action recognition possibility buttons : 
        self.sec_1.ids.with_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        self.sec_1.ids.with_new_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        self.sec_1.ids.floor_fuction_button.pos_hint = {"center_x": 0.5, "center_y": 0.7}
        self.sec_1.ids.stop_button.pos_hint = {"center_x": 0.5, "center_y": 0.3} 
        
        self.msg_with_action_funtion = False
        self.msg_without_action_funtion = True
        self.msg_with_new_action_funtion = False
        
        with_action_pub.publish(self.msg_with_action_funtion)
        without_action_pub.publish(self.msg_without_action_funtion)
        new_action_pub.publish(self.msg_with_new_action_funtion)

        self.floor = True
        cmd_floor = "rosrun ground_plane predict.py"
        self.fl = subprocess.Popen(['xterm', '-e', cmd_floor])
        self.sec_1.ids.floor_fuction_button.disabled = True

        self.sec_1.ids.human_detected.text = ''
        
     
    def new_action_function(self, *args):
        print("new_action_function button_pressed")
        
        # to get rid of other action recognition possibility buttons : 
        self.sec_1.ids.with_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        self.sec_1.ids.floor_fuction_button.pos_hint = {"center_x": 0.5, "center_y": 50}      
        self.sec_1.ids.with_new_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 0.7}
        self.sec_1.ids.stop_button.pos_hint = {"center_x": 0.5, "center_y": 0.3}       
        
        self.msg_with_action_funtion = False
        self.msg_without_action_funtion = False
        self.msg_with_new_action_funtion = True
        
        with_action_pub.publish(self.msg_with_action_funtion)
        without_action_pub.publish(self.msg_without_action_funtion)
        new_action_pub.publish(self.msg_with_new_action_funtion)
        
        # self.event_with_new_action = Clock.schedule_interval(self.listener_with_new_action,  0.5)      
        # self.event_start_stop = Clock.schedule_interval(self.start_stop, 0.5)        
        # trying new things : 
        # to start a new terminal for the python application 
        # not working : display will be connected to Xavier which will use Linux, the application to be started should start/stop in TP!!!
            
    def stop_function(self, *args):
        print("stop button_pressed")
        # self.stop_pressed = True               
        
        # to bring the action recognition options (buttons) back :
        self.sec_1.ids.with_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 0.7}
        self.sec_1.ids.floor_fuction_button.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.sec_1.ids.with_new_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 0.3}                
        self.sec_1.ids.stop_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        
        self.msg_with_action_funtion = False
        self.msg_without_action_funtion = False
        self.msg_with_new_action_funtion = False
        
        stop_pub.publish(self.msg_with_action_funtion)        
        with_action_pub.publish(self.msg_with_action_funtion)
        without_action_pub.publish(self.msg_without_action_funtion)
        new_action_pub.publish(self.msg_with_new_action_funtion)
                
        # To stop subscribing the topics from action detection program
        Clock.unschedule(self.listener_with_action)

        # self.terminate()
        if self.action_rec == True :
            self.sec_1.ids.with_action_function_button.disabled = False
            self.p.terminate() 
        if self.floor == True :
            self.sec_1.ids.floor_fuction_button.disabled = False
            self.fl.terminate() 
                       
     
    def emergency_exit_function(self, *args):
        print("emergency_exit button_pressed")
        
        # To bring the action recognition options (buttons) back :
        self.sec_1.ids.with_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 0.7}
        self.sec_1.ids.floor_fuction_button.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.sec_1.ids.with_new_action_function_button.pos_hint = {"center_x": 0.5, "center_y": 0.3}                                                
        self.sec_1.ids.stop_button.pos_hint = {"center_x": 0.5, "center_y": 50}
        
        self.msg_with_action_funtion = False
        self.msg_without_action_funtion = False
        self.msg_with_new_action_funtion = False
        
        with_action_pub.publish(self.msg_with_action_funtion)
        without_action_pub.publish(self.msg_without_action_funtion)
        new_action_pub.publish(self.msg_with_new_action_funtion)     

        self.terminate()
        
    def Network_quality(self,dt):
    
        results = subprocess.check_output(["netsh", "wlan", "show", "interfaces"])
        encoding='latin1'
        decoded = results.decode(encoding)
        lines = decoded.split('\r\n')
        d = {}
        
        for line in lines:
            if ':' in line:
                vals = line.split(':')
                if vals[0].strip() != '' and vals[1].strip() != '':
                    d[vals[0].strip()] =  vals[1].strip()

        # could be important to check if the wlan NAME exists in directory (if we will use EDAG-Guestt or TP_....) 
        # What happens if connected to multiple networks? 
        # Which signal would be given as output? 
        
        for key in d.keys():
            if key == "Signal":
                # print(d[key])   
                self.sec_1.ids.network.text = "Network Connection"
                self.sec_1.ids.new_work_con.text = 'Signal Quality : ' + d[key]
                
                network_quality = d[key]
                network_quality = network_quality.replace("%","")
                                
                if int(network_quality) >= 80 :
                    self.sec_1.ids.network_im.source = "network_quality/5.png"
                elif int(network_quality) >= 60 and int(network_quality) < 80 :
                    self.sec_1.ids.network_im.source = "network_quality/4.png"
                elif int(network_quality) >= 40 and int(network_quality) < 60 :
                    self.sec_1.ids.network_im.source = "network_quality/3.png"   
                elif int(network_quality) >= 20 and int(network_quality) < 40 :
                    self.sec_1.ids.network_im.source = "network_quality/2.png"
                elif int(network_quality) >= 0 and int(network_quality) < 20 :
                    self.sec_1.ids.network_im.source = "network_quality/1.png"  
                        
    
        '''
        a = psutil.net_if_stats()
        p = psutil.net_io_counters
        print(p)
    
        for network_name in a:
            if network_name == 'Loopback Pseudo-Interface 1':
                print(a[network_name][2])
                # NIC speed expressed in mega bits (MB)
                self.sec_1.ids.new_work_con.text = 'TP NIC Speed : ' + str(a[network_name][2]) + ' MB'
        '''
        
            
    def listener_with_action(self,dt):
        self.topics = rospy.get_published_topics()  

        for i in range (0,len(self.topics)):
            if self.topics[i][0] == "/human_detected" : 
                try:
                    msg_human_detected = rospy.wait_for_message("human_detected", Bool, timeout=1) # only subscribes once and quits
                    msg_human_logged_in = rospy.wait_for_message("human_logged_in", Bool, timeout=1) # only subscribes once and quits

                    if (msg_human_logged_in.data == True) and (msg_human_detected.data == True) : 
                        # self.kv.ids.human_detected.text = '[color=#ff0000]HUMAN LOGGED[/color] IN \n Hands [color=#00ff00]Up To [/color] Surrender '      
                        self.sec_1.ids.human_detected.text = '[font=fonts/OpenSans-Bold.ttf]HUMAN LOGGED IN[/font] \n Hands Up To Surrender '
                    elif (msg_human_logged_in.data == False) and (msg_human_detected.data == True) : 
                        self.sec_1.ids.human_detected.text = '[font=fonts/OpenSans-Bold.ttf]HUMAN DETECTED[/font] \n Wave To Be Logged In'
                    else :
                        self.sec_1.ids.human_detected.text = 'Searching for Human ... '  
                except :
                    self.sec_1.ids.human_detected.text = '' 
                    print('human_detected empty, dont fool me!')
                    pass

    def battery(self,dt):
        self.topics = rospy.get_published_topics()  

        for i in range (0,len(self.topics)):
            if self.topics[i][0] == "/lipo_soc" :     
                try :
                    msg = rospy.wait_for_message("lipo_soc", Float64, timeout=1) # only subscribes once and quits

                    if int(msg.data) >= 80 :
                        self.sec_1.ids.battery_im.source = "battery/5.png"
                    elif int(msg.data) >= 60 and int(msg.data) < 80 :
                        self.sec_1.ids.battery_im.source = "battery/4.png"
                    elif int(msg.data) >= 40 and int(msg.data) < 60 :
                        self.sec_1.ids.battery_im.source = "battery/3.png"   
                    elif int(msg.data) >= 20 and int(msg.data) < 40 :
                        self.sec_1.ids.battery_im.source = "battery/2.png"
                    elif int(msg.data) >= 0 and int(msg.data) < 20 :
                        self.sec_1.ids.battery_im.source = "battery/1.png"       

                    self.sec_1.ids.battery.text = str(int(msg.data)) + '%'
                except :
                    print('lipo topic empty, dont fool me')
                    pass

    def terminate(self):
        if self.manuel == True:
            self.man_1.terminate()
            self.man_2.terminate()
            # self.man_3.terminate()
            # self.man_4.terminate()
            self.man_5.terminate()
            self.manuel = False
            
        if self.autonomus == True:
            self.aut_1.terminate()
            self.aut_2.terminate()
            # self.aut_3.terminate()
            # self.aut_4.terminate()
            # self.aut_5.terminate()
            self.aut_6.terminate()
            # self.aut_7.terminate()
            self.autonomus = False

        if self.action_rec == True :
            self.sec_1.ids.with_action_function_button.disabled = False
            self.p.terminate() 
            self.action_rec = False 

        if self.floor == True :
            self.sec_1.ids.floor_fuction_button.disabled = False
            self.fl.terminate() 

    def on_stop(self): 
        # Execute while killing the whole programm
        # Terminate all subprocesses
        self.terminate()  



if __name__ == "__main__":

    # publisher funtions : 
    with_action_pub = rospy.Publisher('/with_action_pub', Bool, queue_size = 1)
    with_action_pub_2 = rospy.Publisher('/with_action_pub_2', Bool, queue_size = 1)
    without_action_pub = rospy.Publisher('/without_action_pub', Bool, queue_size = 1)
    new_action_pub = rospy.Publisher('/new_action_pub', Bool, queue_size = 1)
    stop_pub = rospy.Publisher('/stop_pub', Bool, queue_size = 1)
          
    rospy.init_node('talker', anonymous=True)
    TutorialApp().run()
  
    
    


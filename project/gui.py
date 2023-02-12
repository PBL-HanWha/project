# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                           0000+
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import threading
#import evdev
#import datetime
from tkinter import *
#from multiprocessing import Process
from threading import Thread
import time
import datetime
#import cv2
import argparse
import os
import platform
import sys
from pathlib import Path
#import torch
#import motor
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, smart_inference_mode
tk = Tk()
angle_x = 0
angle_y = 0

motor_angle_x = 0
motor_angle_y = 0

mode = 4

user_input_x = 0
user_input_y = 0 

line = ''

now_angle_x = 0 # 현재의 x각도 값을 계속 갱신
now_angle_y = 0  # 현재의 x각도 값을 계속 갱신

count=0

x_ok = 0
y_ok = 0


x =''
y =''
offset_string='1231241243'
text = StringVar()
## For 조이스틱 
# anglex=0
# angley=0
# before=None
# Isstopped=False
# devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
# gampepad=None
# joystick_flag=0

## offset
offset_flag = 0
offset_x = 0 
offset_y = 0
enable = 1

##
turnback_flag = 0




def gui():
    global tk,mode, angle_x, angle_y, line,  now_angle_x , now_angle_y, anglex,angley,gamepad,Isstopped,before,devices,turnback_flag,offset_string
    
    def refresh():
        o = 'OFFSET : (' + str(1)+ ' , ' + str(2) + ')'
        lab.config(text=o)
        lab['text'] = o
        tk.after(1000, refresh) # run itself again after 1000 ms



    tk.title('Hanwha Systems PBL - Naval Gun Control') # GUI 창 제목 설정
    # tk.geometry("800x760")
    label1 = Label(tk, text='----------------------------------------[[Set Angle]]------------------------------------').grid(row=0, column=0, columnspan=3, sticky=EW) # row,column = 0,0위치에 3칸짜리(column span) ,East West쪽으로 쭉 늘림 (Sticky)
    label2 = Label(tk, text='x_angle').grid(row=1, column=0) # 1,0 위치에 x_angle
    label3 = Label(tk, text='y_angle').grid(row=2, column=0) # 2,0 위치에 y_angle
	#label3 = Label(tk, text='y_angle').grid(row=8, column=0)
	#label4 = Label(tk, text='y_angle').grid(row=2, column=0)
	#label6 = Label(tk, text='').grid(row=9, column=0, columnspan=3, sticky=EW) # 공백 
    label4 = Label(tk, text='----------------------------------------[[Mode Selection]]------------------------------------').grid(row=3, column=0, columnspan=3,sticky=EW)
    label5 = Label(tk, text='Nvidia Jetson Nano 4GB Development Kit (JetPack 4.6)').grid(row=7, column=0, columnspan=3, sticky=EW)
	#label3 = Label(tk, text='y_angle').grid(row=8, column=0)
	#label4 = Label(tk, text='y_angle').grid(row=2, column=0)
    lab = Label(tk, fg = 'red')
    lab.grid(row=2, column=2, sticky=NS)


    entry1 = Entry(tk) # 입력칸 선언 
    entry2 = Entry(tk) # 입력칸 선언 

    entry1.grid(row=1, column=1) # 1,1 위치에 빈칸 생성 (x_angle 입력할 곳)     
    entry2.grid(row=2, column=1) # 2,1 위치에 빈칸 생성 (y_angle 입력할 곳) 
                                         
                                               

    def default():
        global now_angle_x, now_angle_y, offset_flag
        pass
         

                                              
    def offset_mode():
        global offset_flag
        if offset_flag == 0 : 
            offset_flag = 1
            print("******Offset Correction ON******\n")
        else : 
            offset_flag = 0
            print("******Offset Correction OFF******\n")

    def mode1():
        global mode, motor_angle_x, motor_angle_y, now_angle_x, now_angle_y, offset_x, offset_y, turnback_flag,offset_flag,x_ok,y_ok
        pass

    def turn(): #수동으로 회전할 때 쓰는 함수 
        global mode, angle_x, angle_y, now_angle_x, now_angle_y , offset_x, offset_y , motor_angle_x, motor_angle_y, turnback_flag,offset_flag
        pass

        mode = 1 # 수동으로 입력한 각도에 도달한 이후엔 Tracking (mode=1)

    def patrol(): # 전체 훑을 때 쓰는 함수
        global mode, line, now_angle_x, now_angle_y, offset_flag, enable
        pass

    def joystick():  # 조이스틱 제어를 위한 함수 
        global anglex,angley,gamepad,Isstopped,before,devices,mode, motor_angle_x, motor_angle_y, now_angle_x, now_angle_y,offset_flag,joystick_flag
        mode=3
        pass
    


    button1 = Button(tk, text='Enter', bg='black', fg='white', width=20, command=turn).grid(row=1, column=2, sticky=NS)
    # Enter버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = turn , 여기선 turn이란 함수 실행


    button2 = Button(tk, text='Patrol', bg='black', fg='white', width=20, command=patrol).grid(row=4, column=0,sticky=S)
    # Patrol 모드 버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = patrol , 여기선 patrol이란 함수 실행



    button3 = Button(tk, text='Tracking', bg='black', fg='white', width=20, command=mode1).grid(row=4, column=1,sticky=S)
    # Traking버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = mode1 , 여기선 mode1이란 함수 실행
    # mode는 global 변수이니까 값을 전체 함수가 공유함 ! -> mode1이라는 함수는 mode 변수를 1로 만들어주는 함수이므로
    # 이 버튼을 누르면 mode값이 1이 되면서 돌고있는 model 내에서의 동작도 달라지게 됨 
    # model 함수 내에서 if문으로 mode별 동작을 다르게했었으니까


    button4 = Button(tk, text='Control', bg='black', fg='white', width=20, command=joystick).grid(row=4, column=2, sticky=S)
    # Control버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = joystick , 여기선 joystick이란 함수 실행

    button5 = Button(tk, text='Offset mode', bg='black', fg='white', width=20, command=offset_mode).grid(row=5, column=1, sticky=S)
    button6 = Button(tk, text='Offset Reset', bg='black', fg='white', width=20, command=offset_mode).grid(row=5, column=2, sticky=S)

    button7 = Button(tk, text='Default', bg='black', fg='white', width=20, command=default).grid(row=5, column=0, sticky=S)

    refresh()
    tk.mainloop() # GUI가 계속 입력값을 다시 받을 수 있도록 Loop
 
def test():
    global offset_string , count
    while True :
        count += 1
        print(count)
        offset_string = "offset : " + str(count)

t0 = Thread (target = gui())
t0.start()







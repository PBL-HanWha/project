#!/usr/bin/env python3

#coding=utf-8


import time
from Arm_Lib import Arm_Device
Arm = Arm_Device()
def movex(id, angle, offset_x,rotation_time = 500):
    # id가 1,2 일 때 가능
    if angle >180 : angle = 180    # 최대 및 최소 각도 설정 (서보모터 자체 제한)
    if angle <0 : angle = 0
    angle+=offset_x
    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)
    return angle

def movey(id, angle,offset_y, rotation_time = 500):
    # id가 1,2일 때 가능

    angle = angle + 90
    if angle >125 : angle = 125 # 최대 및 최소 각도 설정 (0~180으로 하면 Frame 밖의 배경이 보일까봐 제한함)
    if angle <80 : angle = 80
    
    angle+=offset_y
    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)
    return angle-90

# max,min_normalization_range 이건 안씀 
def normalize_0_to_180_x(original_number, max_range = 640, min_range = 45, max_normalization_range = 135, min_normalization_range = 0):
    normalized_num = round( (original_number-320)/33)
    # when camera moves 2 degrees, almost 31px moving
    
    return normalized_num


    
# max,min_normalization_range 이건 안씀 
def normalize_0_to_180_y(original_number, max_range = 480, min_range = 0, max_normalization_range = 35, min_normalization_range = 0):
    normalized_num = round( (original_number-240)/32 )
    # when camera moves 2 degrees, almost 22px moving 

    return normalized_num


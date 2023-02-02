#!/usr/bin/env python3

#coding=utf-8


import time
from Arm_Lib import Arm_Device

def movex(id, angle, rotation_time = 500):
    # id가 1,2,3,4,6 일 때 가능
    # id가 5일 경우 동작범위가 0 ~ 270 까지 이므로, normalize_0_to_180의 max_normalization_range를 270으로 설정해야됨
    Arm = Arm_Device()
    if angle >135 : angle = 135
    if angle <45 : angle = 45

    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)

def movey(id, angle, rotation_time = 500):
    # id가 1,2,3,4,6 일 때 가능
    # id가 5일 경우 동작범위가 0 ~ 270 까지 이므로, normalize_0_to_180의 max_normalization_range를 270으로 설정해야됨
    Arm = Arm_Device()
    if angle >35 : angle = 35
    if angle <0 : angle = 0

    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)


def move_exit(id, angle, rotation_time = 500):
    # id가 1,2,3,4,6 일 때 가능
    # id가 5일 경우 동작범위가 0 ~ 270 까지 이므로, normalize_0_to_180의 max_normalization_range를 270으로 설정해야됨
    Arm = Arm_Device()
    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)
    exit()


def normalize_0_to_180_x(original_number, max_range = 640, min_range = 45, max_normalization_range = 135, min_normalization_range = 0):
    normalized_num = round( (original_number-320)/ (max_range - min_range) *45 )
    
    #if normalized_num > max_normalization_range : normalized_num = max_normalization_range
    #if normalized_num < min_normalization_range : normalized_num = min_normalization_range 
    return normalized_num

def normalize_0_to_180_y(original_number, max_range = 480, min_range = 0, max_normalization_range = 35, min_normalization_range = 0):
    normalized_num = round( (original_number-240)/ (max_range - min_range) *45 )
    #if normalized_num > max_normalization_range : normalized_num = max_normalization_range
    #if normalized_num < min_normalization_range : normalized_num = min_normalization_range 
    return normalized_num



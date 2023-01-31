#!/usr/bin/env python3

#coding=utf-8

import time
from Arm_Lib import Arm_Device

def move(id, angle, rotation_time = 500):
    # id가 1,2,3,4,6 일 때 가능
    # id가 5일 경우 동작범위가 0 ~ 270 까지 이므로, normalize_0_to_180의 max_normalization_range를 270으로 설정해야됨
    Arm = Arm_Device()
    time.sleep(0.001)
    Arm.Arm_serial_servo_write(id, angle, rotation_time)
    time.sleep(0.001)

    print(Arm.Arm_serial_servo_read(id))


def normalize_0_to_180(original_number, max_range = 640, min_range = 0, max_normalization_range = 180, min_normalization_range = 0):
    normalized_num = round(original_number * (max_normalization_range - min_normalization_range) / (max_range - min_range))
    
    return normalized_num


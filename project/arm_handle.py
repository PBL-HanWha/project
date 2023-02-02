#!/usr/bin/env python

import evdev
import motor
import time

angle=90
motor.move(3,90)
motor.move(4,90)
ids=0
now=0


devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
gampepad=None

for device in devices:
	if device.name==' USB Gamepad          ':
		gamepad=evdev.InputDevice(device.path)
		while True:
			start_time=time.time()
			tp=0

			print('restart')

			for event in gamepad.read_loop():
				if (now-start_time<2):
					now=time.time()
				else:
					break
				print(event)
				if event.type == evdev.ecodes.EV_KEY and event.value==1: #right joystick on state
					ids=4
					if event.code==289 or event.code==290:
						angle+=10
						motor.move(ids,angle)
					elif event.code==291 or event.code==288:
						angle-=10
						motor.move(ids,angle)
				if event.type == 3 and event.value!=127: #left joystick on state
					ids=3
					if event.value==0:
						angle+=1
						motor.move(ids,angle)
					elif event.value==255:
						angle-=1
						motor.move(ids,angle) 



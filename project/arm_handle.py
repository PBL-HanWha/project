#!/usr/bin/env python

import evdev
import motor
import time

angle=90
motor.movex(1,90)
motor.movex(2,90)
before=None


devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
gampepad=None

for device in devices:
	if device.name==' USB Gamepad          ':
		gamepad=evdev.InputDevice(device.path)
		while True:
			event=gamepad.read_one()
			if event==None:
				if before!=None:
					print('no')
					event=before

			elif event.value==127:
				print('stop')
				before='stop'
			if event and before!='stop':
				if event.type == evdev.ecodes.EV_KEY and event.value==1: #right joystick on state
					before=event
					ids=1
					if event.code==289 or event.code==290:
						print('id1 up')
						angle+=1
						motor.movex(ids,angle)
					elif event.code==291 or event.code==288:
						print('id1 down')
						angle-=1
						motor.movex(ids,angle)
				if event.type == 3 and event.value!=127: #left joystick on state
					before=event	
					ids=2
					if event.value==0:
						angle+=1
						motor.movex(ids,angle)
						time.sleep(0.1)
					elif event.value==255:
						angle-=1
						motor.movex(ids,angle)
						time.sleep(0.1)
				'''if event.type == evdev.ecodes.EV_KEY and event.value==1: #right joystick on state
					ids=4
					if event.code==289 or event.code==290:
						angle+=10
						motor.move(ids,angle)
					elif event.code==291 or event.code==288:
						angle-=10
						motor.move(ids,angle)
					print(event.type)
				if event.type == 3 and event.value!=127: #left joystick on state
					ids=3
					if event.value==0:
						angle+=1
						motor.move(ids,angle)
					elif event.value==255:
						angle-=1
						motor.move(ids,angle)
						
			for event in gamepad.read_loop():
				print(event)
				if event.type == evdev.ecodes.EV_KEY and event.value==1: #right joystick on state
					ids=4
					if event.code==289 or event.code==290:
						start_time=time.time()
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
						motor.move(ids,angle) '''



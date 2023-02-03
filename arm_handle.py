#!/usr/bin/env python

import evdev
import motor
import time

anglex=90
angley=0
motor.movex(1,90)
motor.movey(2,0)
before=None
Isstopped=False


devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
gampepad=None

def setanglex(anglex):
	if anglex >135 : anglex = 135
	elif anglex <45 : anglex = 45

def setangley(angley):
	if angley >35 : angley = 35
	elif angley <0 : angley = 0

for device in devices:
	if device.name==' USB Gamepad          ':
		gamepad=evdev.InputDevice(device.path)
		while True:
			time.sleep(.08)
			event=gamepad.read_one()
			if event==None and Isstopped==False:
				if before!=None:
					event=before
					Isstopped=False

			if event:
				if event.type == evdev.ecodes.EV_KEY:
					if event.value==0:
						Isstopped=True
					if event.value==1: #right joystick on state
						Isstopped=False
						before=event
						ids=1
						if event.code==289 or event.code==290:
							anglex+=1
							if anglex >135 : 
								anglex = 135
							elif anglex <45 : 
								anglex = 45
							print(anglex)
							motor.movex(ids,anglex)
						elif event.code==291 or event.code==288:
							anglex-=1
							if anglex >135 : 
								anglex = 135
							elif anglex <45 : 
								anglex = 45
							print(anglex)
							motor.movex(ids,anglex)
				if event.type == 3:
					if event.value==127:
						print('stop')
						Isstopped=True
					elif event.value!=127: #left joystick on state
						Isstopped=False
						before=event	
						ids=2
						if event.value==0:
							angley+=1
							if angley >35 : 
								angley = 35
							elif angley <0 : 
								angley = 0
							print(angley)
							motor.movey(ids,angley)
						elif event.value==255:
							angley-=1
							if angley >35 : 
								angley = 35
							elif angley <0 : 
								angley = 0
							print(angley)
							motor.movey(ids,angley)
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


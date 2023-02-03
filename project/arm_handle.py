#!/usr/bin/env python

import evdev
import motor
import time

anglex=90
angley=0
before=None
Isstopped=False


devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
gampepad=None


motor.movex(1,90)
motor.movey(2,0)

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
						ids=2
						if event.code==289 or event.code==288:
							angley+=1
							if angley >35 : 
								angley = 35
							elif angley <0 : 
								angley = 0
							motor.movey(ids,angley)
						elif event.code==291 or event.code==290:
							angley-=1
							if angley >35 : 
								angley = 35
							elif angley <0 : 
								angley = 0
							motor.movey(ids,angley)
				if event.type == 3:
					if event.value==127:
						Isstopped=True
					elif event.value!=127: #left joystick on state
						Isstopped=False
						before=event	
						ids=1
						if event.value==0:
							anglex-=1
							if anglex >135 : 
								anglex = 135
							elif anglex <45 : 
								anglex = 45
							motor.movex(ids,anglex)
						elif event.value==255:
							anglex+=1
							if anglex >135 : 
								anglex = 135
							elif anglex <45 : 
								anglex = 0
							motor.movex(ids,anglex)
				

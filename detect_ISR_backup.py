from threading import Thread 
import os
import RPi.GPIO as GPIO

        ##

ISR_pin = 18 #interruptSW pin : 18
def ISR_init():
    GPIO.setmode(GPIO.BOARD)  
    GPIO.setup(ISR_pin, GPIO.IN)   
    
def ISR_intr_SW():  
    GPIO.add_event_detect(ISR_pin, GPIO.FALLING, callback=blink, bouncetime=10)    
    try:
        while True:
            pass
    finally:
        GPIO.cleanup()         

def GUI() :
    pass

        ##

def manual_mode():
    pass

mode = 0 #0:exception(top menu), 1:auto, 2:manual
def main():

    GUI_thread = Thread(target=GUI, args=(), kwargs={})    
    GUI_thread.start()
    ISR_init()
    #...setup       
    
    while(True):
        if(mode==0):
            print("program ready\n")

        elif(mode==1):
            pass
        #...loop    

        elif(mode==2):
            manual_mode_thread = Thread(target=manual_mode, args=(), kwargs={})    
            manual_mode_thread.start()
            manual_mode_thread.join()
        else:
            print("error")    
    


if __name__ == '__main__':
    main()
   
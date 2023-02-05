from tkinter import *
import motor
import time


def gui():
    global mode, angle_x, angle_y, line,  now_angle_x , now_angle_y, anglex,angley,gamepad,Isstopped,before
    tk = Tk()

    tk.title('Motor Control') # GUI 창 제목 설정
    # tk.geometry("800x760")
    label0 = Label(tk,
                   text='--------------------------------------------------[[Set Angle]]----------------------------------------------').grid(
        row=0, column=0, columnspan=3, sticky=EW) # row,column = 0,0위치에 3칸짜리(column span) ,East West쪽으로 쭉 늘림 (Sticky)
    label1 = Label(tk, text='x_angle').grid(row=1, column=0) # 1,0 위치에 x_angle
    label2 = Label(tk, text='y_angle').grid(row=2, column=0) # 2,0 위치에 y_angle
    label_space = Label(tk, text='').grid(row=3, column=0, columnspan=3, sticky=EW) # 공백 

    label3 = Label(tk,
                   text='-----------------------------------------------[[Mode Selection]]-------------------------------------------').grid(
        row=4, column=0, columnspan=3, sticky=EW) # row,column = 0,3 위치에 3칸짜리(column span) ,East West쪽으로 쭉 늘림 (Sticky)

    label_space = Label(tk, text='').grid(row=10, column=0, columnspan=3, sticky=EW) # 공백 


    label7 = Label(tk, text='-----------------------------------------------[[Priority Setting]]-------------------------------------------').grid(row=11, column=0, columnspan=3,sticky=EW)

    label6 = Label(tk, text='').grid(row=12, column=0, columnspan=3, sticky=EW) # 공백 
    label4 = Label(tk, text='Nvidia Jetson Nano 4GB Development Kit (JetPack 4.6)').grid(row=13, column=0, columnspan=3,
                                                                                         sticky=EW)
    label5 = Label(tk, text='CUDA 10.2  / OpenCV 4.5.3 with CUDA / PyTorch 1.8.0 with CUDA / Torchvision 0.9.0').grid(
        row=14, column=0, columnspan=3, sticky=EW)





    entry1 = Entry(tk) # 입력칸 선언 
    entry2 = Entry(tk) # 입력칸 선언 

    entry1.grid(row=1, column=1) # 1,1 위치에 빈칸 생성 (x_angle 입력할 곳) 
    entry2.grid(row=2, column=1) # 2,1 위치에 빈칸 생성 (y_angle 입력할 곳)

    

    def joystick():  # 조이스틱 제어를 위한 함수 
        global anglex,angley,gamepad,Isstopped,before
        for device in devices:
                            if device.name==' USB Gamepad          ':
                                gamepad=evdev.InputDevice(device.path)
                                while True:
                                    time.sleep(.08) # 이거 작게 줄수록 조이스틱 조작에 민감함 
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
                                                    #print(anglex)
                                                    
                                                    if mode==3 : motor.movex(ids,anglex)
                                                elif event.code==291 or event.code==288:
                                                    anglex-=1
                                                    if anglex >135 : 
                                                        anglex = 135
                                                    elif anglex <45 : 
                                                        anglex = 45
                                                    #print(anglex)
                                                    if mode==3 : motor.movex(ids,anglex)
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
                                                    #print(angley)
                                                    if mode==3 : motor.movey(ids,angley)
                                                elif event.value==255:
                                                    angley-=1
                                                    if angley >35 : 
                                                        angley = 35
                                                    elif angley <0 : 
                                                        angley = 0
                                                    #print(angley)
                                                    if mode==3 : motor.movey(ids,angley)
    def mode1():
        global mode,motor_angle_x,motor_angle_y,now_angle_x,now_angle_y
        mode = 1
        motor.movex(1, now_angle_x - motor_angle_x) # 현재 각도에 x화소 값 차이만큼 이동
        motor.movey(2, now_angle_y + motor_angle_y) # 현재 각도에 y화소 값 차이만큼 이동 

                        #  now_x - motor_x 여기만 뺄셈인 이유는 모터 회전방향 때문임 
                        #  x화소는 좌측에서 우측으로 갈 수록 증가하지만
                        #  모터가 0도일때 오른쪽을 보고 180도일때 왼쪽을 봐서 서로 반대임

  

        now_angle_x = now_angle_x - motor_angle_x #
        now_angle_y = now_angle_y + motor_angle_y # 돌아간 이후에 현재 각도 값을 갱신
        time.sleep(3)

    def turn(): #수동으로 회전할 때 쓰는 함수 
        global mode, angle_x, angle_y, now_angle_x, now_angle_y
        mode = 0  # 수동으로 각도 변환 중에 Detect가 이루어지면 안됨 --> Turning... 을 출력하고 아무것도 안하는 상태인 mode=0으로 설정 
        angle_x = int(entry1.get()) # 빈칸으로 받은 값 int 저장
        angle_y = int(entry2.get()) # 빈칸으로 받은 값 int 저장

        motor.movex(1, angle_x) # motor.move가 movex와 movey로 나눠진 이유 -->> x는 0~180도 y는 0~90도로 제한되니까 상한하한 따로 처리하기 위해 분리함
        motor.movey(2, angle_y)

        now_angle_x = angle_x # 각도 돌린 후에 현재 각도 값 갱신
        now_angle_y = angle_y

        mode = 1 # 수동으로 입력한 각도에 도달한 이후엔 Tracking (mode=1)

    def patrol(): # 전체 훑을 때 쓰는 함수
        global mode, line, now_angle_x, now_angle_y
        mode = 2
        for i in range(180): # x값은 0~180 천천히 이동하고 y값 조금씩 증가시켜서 전체 map을 훑는다
            now_angle_x = i # 현재 각도 값 갱신
            now_angle_y = 0 # 현재 각도 값 갱신
            motor.movey(2, 0)
            motor.movex(1, i)
            time.sleep(0.05)
        for i in range(180, 1, -1):
            now_angle_x = i # 현재 각도 값 갱신
            now_angle_y = 10 # 현재 각도 값 갱신 
            motor.movey(2, 10)
            motor.movex(1, i)
            time.sleep(0.05)
        for i in range(180):
            now_angle_x = i # 현재 각도 값 갱신
            now_angle_y = 20 # 현재 각도 값 갱신
            motor.movey(2, 20)
            motor.movex(1, i)
            time.sleep(0.05)
        for i in range(180, 1, -1):
            now_angle_x = i # 현재 각도 값 갱신
            now_angle_y = 30 # 현재 각도 값 갱신
            motor.movey(2, 30)
            motor.movex(1, i)
            time.sleep(0.05)

        # 이렇게 motor 도는 동안 yolo모델 함수는 계속 돌고 있음 ! 
        # 일정 각도 내로 들어오면 log.txt파일 열어서 관측결과 저장하는 중

        time.sleep(.5) # 혹시 저장 덜 됐을지 모르니 좀 기다려 줌 
        motor.movex(1,90)
        motor.movey(2,0) #
        now_angle_x = 90
        now_angle_y = 0




        

        f = open('log.txt', 'r')  # Detect결과 저장된 txt파일 읽어오기
        line = f.read() # 다 읽어옴 
        f.close()


        # 아래 코드는 txt파일 새로운 gui창으로 여는 코드 
        root = Tk()
        widget = Text(root)
        scrollbar = Scrollbar(root)
        scrollbar.pack(side=RIGHT, fill=Y)
        widget.pack(side=LEFT, fill=Y)
        scrollbar.config(command=widget.yview)
        widget.config(yscrollcommand=scrollbar.set)
        widget.insert(END, line)

        os.remove('log.txt') # 다 본 후에는 지워줌 

        motor.movex(1, 0)
        motor.movey(2, 0)  # 제자리 정렬

        now_angle_x = 0 # 현재 각도 값 갱신
        now_angle_y = 0

        mode = 4

    button1 = Button(tk, text='Enter', bg='black', fg='white', width=20, command=turn).grid(row=1, column=2, rowspan=2,
                                                                                            sticky=NS)
    # Enter버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = turn , 여기선 turn이란 함수 실행


    button2 = Button(tk, text='Patrol', bg='black', fg='white', width=20, command=patrol).grid(row=5, column=0,
                                                                                               rowspan=2, sticky=S)
    # Patrol 모드 버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = patrol , 여기선 patrol이란 함수 실행



    button3 = Button(tk, text='Tracking', bg='black', fg='white', width=20, command=mode1).grid(row=5, column=1,
                                                                                                rowspan=2, sticky=S)
    # Traking버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = mode1 , 여기선 mode1이란 함수 실행
    # mode는 global 변수이니까 값을 전체 함수가 공유함 ! -> mode1이라는 함수는 mode 변수를 1로 만들어주는 함수이므로
    # 이 버튼을 누르면 mode값이 1이 되면서 돌고있는 model 내에서의 동작도 달라지게 됨 
    # model 함수 내에서 if문으로 mode별 동작을 다르게했었으니까


    button4 = Button(tk, text='Control', bg='black', fg='white', width=20, command=joystick).grid(row=5, column=2,
                                                                                              rowspan=2, sticky=S)
    # Control버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = joystick , 여기선 joystick이란 함수 실행

    button5 = Button(tk, text='Ship1', bg='black', fg='white', width=20, command=joystick).grid(row=12, column=1,
                                                                                               sticky=S)
    button6 = Button(tk, text='Ship2', bg='black', fg='white', width=20, command=joystick).grid(row=12, column=2,
                                                                                               sticky=S)
    button7 = Button(tk, text='Airplane1', bg='black', fg='white', width=20, command=joystick).grid(row=13, column=1,
                                                                                               sticky=S)
    button8 = Button(tk, text='Airplane2', bg='black', fg='white', width=20, command=joystick).grid(row=13, column=2,
                                                                                               sticky=S)

    label_ship = Label(tk, text='[Ship List]').grid(row=12, column=0, sticky=EW)
    label_air = Label(tk, text='[Airplane List]').grid(row=13, column=0, sticky=EW)
    label_space = Label(tk, text='').grid(row=14, column=0, columnspan=3, sticky=EW) # 공백 



    tk.mainloop() # GUI가 계속 입력값을 다시 받을 수 있도록 Loop




gui()

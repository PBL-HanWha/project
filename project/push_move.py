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
import evdev
import datetime
from tkinter import *
from multiprocessing import Process
from threading import Thread
import time
import cv2
import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import motor
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

motor_angle_x = 90
motor_angle_y = 0

mode = 4

user_input_x = 0
user_input_y = 0 

line = ''

now_angle_x = 90 # 현재의 x각도 값을 계속 갱신
now_angle_y = 0  # 현재의 x각도 값을 계속 갱신

count=0

## For 조이스틱 
anglex=90 
angley=0
before=None
Isstopped=False
devices=[evdev.InputDevice(path) for path in evdev.list_devices()]
gampepad=None

def setanglex(anglex): # 조이스틱 조작에 쓰이는 함수
	if anglex >135 : anglex = 135
	elif anglex <45 : anglex = 45

def setangley(angley): # 조이스틱 조작에 쓰이는 함수
	if angley >35 : angley = 35
	elif angley <0 : angley = 0




@smart_inference_mode()
def runn(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global motor_angle_x, motor_angle_y, mode, angle_x, angle_y, line,  now_angle_x, now_angle_y,count

    motor.movex(1, 90)
    motor.movey(2, 0)  # 초기 정렬

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det[:1]):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2)
                    circle = cv2.circle(im0, center_point, 5, (0, 255, 0), 2)
                    text_coord = cv2.putText(im0, str(center_point), center_point, cv2.FONT_HERSHEY_PLAIN, 2,
                                             (0, 0, 255))

                    center_x = center_point[0]             # 현재 라운딩박스 중심 x좌표
                    center_y = -(center_point[1] - 480)     # 현재 라운딩박스 중심 y좌표

                    # -(y-480)하는 이유 : 코드로직상 좌측 상단이 (0,0) 우측 하단이 (640,480)임 
                    count +=1
                  


                    if now_angle_x > 135 : now_angle_x = 135
                    if now_angle_x < 45   : now_angle_x = 45
                    if now_angle_y > 35 : now_angle_y = 35
                    if now_angle_y < 0   : now_angle_y = 0                


                    print(count)
                    #print(names)
                    #print(names[int(det[0, 5])])
                    
                    print(det)
                    print('vvvvvvvvvvvvvvvv')
                    print(det[:1])

                    print (center_x)
                    print (center_y)

                    motor_angle_x = motor.normalize_0_to_180_x(center_x)
                    motor_angle_y = motor.normalize_0_to_180_y(center_y) # 이 화소값을 기반으로 angle값으로 변환 

                    if (mode == 0):  # mode = 0 만든 이유 -> 수동각도 입력 시 돌아가는 도중에 다른걸 추적하지 않기 위해
                                     # mode = 2인 상태로 유지되고 있으면 정한 각도로 이동하다가 중간에 Detect되면 거기에 잡혀버림
                        print('Turning  . . . . .')  
                        ###############################################################################################################

                    if (mode == 1):
                        print('mode 1 : Tracking Mode')
                        #motor.movex(1, now_angle_x - motor_angle_x) # 현재 각도에 x화소 값 차이만큼 이동
                        #motor.movey(2, now_angle_y + motor_angle_y) # 현재 각도에 y화소 값 차이만큼 이동 

                        #  now_x - motor_x 여기만 뺄셈인 이유는 모터 회전방향 때문임 
                        #  x화소는 좌측에서 우측으로 갈 수록 증가하지만
                        #  모터가 0도일때 오른쪽을 보고 180도일때 왼쪽을 봐서 서로 반대임

                        print ('now_angle_x:')
                        print (now_angle_x)
                        print ('now_angle_y:')
                        print (now_angle_y)


                        #now_angle_x = now_angle_x - motor_angle_x #
                        #now_angle_y = now_angle_y + motor_angle_y # 돌아간 이후에 현재 각도 값을 갱신
                        #time.sleep(3)

                    if (mode == 1 and (abs(center_x - 320) < 15 and abs(center_y - 240) < 15)): # 일정 각도 내에 Detect될 경우
                        time.sleep(5) # 5초 가만히 있다가 
                        motor.movex(1, 0)  # 일정각도로 다시 되돌아간다
                        motor.movey(2, 0)

                        now_angle_x = 0 # 현재 각도 값 갱신 
                        now_angle_y = 0

                        mode = 4 # Idle State로 돌아감 (아무것도 안하는 상태)
                        ###############################################################################################################
                    if (mode == 2): #전체 훑는 모드 --> for문 써야하는데 여기에 for문 넣으니까 yolo가 멈춰서 
                                    #                   이 모드에선 motor.move를 gui에서 처리함  
                                    
                        print('mode 2 : Patrol Mode')

                    if (mode == 2 and (abs(center_x - 320) < 15 and abs(center_y - 240) < 15)): # Patrol 중 일정 각도 내에 들어오면
                        name = 'ship' # name 뭘로 받을지 몰라서 일단 ship으로 해놨었음 
                        print('detected') # 탐지되면 Detected 라는 말을 띄우고
                        f = open('log.txt', 'a')  # log.txt라는 파일을 append (추가) 모드로 생성 및 열기
                        msg = [str(datetime.datetime.now()), ': [', names[int(det[0, 5])], '] is detected... Location : X = ',
                               str(now_angle_x), ', Y = ', str(now_angle_y), '\n'] # 현재 시간 : name , x,y 값을 리스트로 
                        msg = ''.join(msg) # join 함수를 이용해서 이어 붙여서 문자열을 만듦
                        f.write(msg) # 만든 문자열을 log.txt에 작성
                        f.close      
                        ###############################################################################################################
                    if (mode == 3): # 조이스틱 모드 
                        print('mode 3 : Control Mode') # 여기서 제어하니까 카메라가 멈춰서 이것도 GUI 쪽에서 제어 

                        ###############################################################################################################
                    #
                    if (mode == 4): # Idle 상태 : 아무것도 안하는 상태 (대기상태)
                        print('Idle State')
                

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
            #LOGGER.info(f"{s} **** {'' if len(det) else '(no detections), '} ****** {dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def gui():
    global mode, angle_x, angle_y, line,  now_angle_x , now_angle_y, anglex,angley,gamepad,Isstopped,before
    tk = Tk()

    tk.title('Motor Control') # GUI 창 제목 설정
    # tk.geometry("800x760")
    label0 = Label(tk,
                   text='----------------------------------------[[Set Angle]]------------------------------------').grid(
        row=0, column=0, columnspan=3, sticky=EW) # row,column = 0,0위치에 3칸짜리(column span) ,East West쪽으로 쭉 늘림 (Sticky)
    label1 = Label(tk, text='x_angle').grid(row=1, column=0) # 1,0 위치에 x_angle
    label2 = Label(tk, text='y_angle').grid(row=2, column=0) # 2,0 위치에 y_angle
    label3 = Label(tk,
                   text='-------------------------------------[[Mode Selection]]---------------------------------').grid(
        row=3, column=0, columnspan=3, sticky=EW) # row,column = 0,3 위치에 3칸짜리(column span) ,East West쪽으로 쭉 늘림 (Sticky)

    label6 = Label(tk, text='').grid(row=8, column=0, columnspan=3, sticky=EW) # 공백 
    label4 = Label(tk, text='Nvidia Jetson Nano 4GB Development Kit (JetPack 4.6)').grid(row=9, column=0, columnspan=3,
                                                                                         sticky=EW)
    label5 = Label(tk, text='CUDA 10.2  / OpenCV 4.5.3 with CUDA / PyTorch 1.8.0 with CUDA / Torchvision 0.9.0').grid(
        row=10, column=0, columnspan=3, sticky=EW)

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

        time.sleep(4) # 혹시 저장 덜 됐을지 모르니 좀 기다려 줌 

        

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


    button2 = Button(tk, text='Patrol', bg='black', fg='white', width=20, command=patrol).grid(row=4, column=0,
                                                                                               rowspan=2, sticky=S)
    # Patrol 모드 버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = patrol , 여기선 patrol이란 함수 실행



    button3 = Button(tk, text='Tracking', bg='black', fg='white', width=20, command=mode1).grid(row=4, column=1,
                                                                                                rowspan=2, sticky=S)
    # Traking버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = mode1 , 여기선 mode1이란 함수 실행
    # mode는 global 변수이니까 값을 전체 함수가 공유함 ! -> mode1이라는 함수는 mode 변수를 1로 만들어주는 함수이므로
    # 이 버튼을 누르면 mode값이 1이 되면서 돌고있는 model 내에서의 동작도 달라지게 됨 
    # model 함수 내에서 if문으로 mode별 동작을 다르게했었으니까


    button4 = Button(tk, text='Control', bg='black', fg='white', width=20, command=joystick).grid(row=4, column=2,
                                                                                              rowspan=2, sticky=S)
    # Control버튼 ! -> 색깔과 크기를 설정하고, 이 버튼이 눌렸을 때 어떤 함수가 실행되는가 : command = joystick , 여기선 joystick이란 함수 실행



    tk.mainloop() # GUI가 계속 입력값을 다시 받을 수 있도록 Loop
 

def main(opt):
    global motor_angle_x, motor_angle_y, mode, user_input_x, user_input_y
    check_requirements(exclude=('tensorboard', 'thop'))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    t1 = Thread(target=gui)
    t1.start()

    t0 = Thread(target=runn(**vars(opt)))
    t0.start()

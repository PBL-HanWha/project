from threading import Thread 
import os
import RPi.GPIO as GPIO
import motor
import time
import cv2
import argparse
import platform
import sys
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



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

def motor_drive(x,y):
    motor.move(1,x)
    motor.move(2,y)



##


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

mode = 0 #0:exception(top menu), 1:auto, 2:manual
def main(
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

    GUI_thread = Thread(target=GUI, args=(), kwargs={})    
    GUI_thread.start()
    ISR_init()
    #...setup      
    source = str(source) 
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) #device 설정 과 소스?
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    ###
    while(True):
        if(mode==0):
            print("program ready\n")

        elif(mode==1):
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
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1 #웹캠이 선택? or 아무것도 입력이 없을 경우 
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count # im0s는 cls를 판단한 string이 달려나옴. frame은 count에 사실 프레임 내용도 없어도 됨. 
                            s += f'{i}: ' #string을 표시해주는거 같음. i는 path[i]에 포함되는 i임 즉 몇번째 장면에 대한 내용인지 확인하는 내용임.
                        else: #예외처리를 해준거 같음 아닐대는 그냥 기본 사진을 해다 프레임 0처리 해버림
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg #프레임 마다 사진파일로 가져다가 처리해서 사진파일 단위로 잘라서 사용함.
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string #필요없는 부분 같긴함. 
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() #det가 뭘까 결정된 요소의 성질을 가짐. det에 박스를 다시 해서 넣어줌 

                            for *xyxy, conf, cls in reversed(det):
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])) #여기서 사실 박스의 3가지 꼭지점이 나옴. 
                                center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2) #여기서 센터포인트 계산하고 
                                circle = cv2.circle(im0, center_point, 5, (0,255,0), 2)
                                text_coord = cv2.putText(im0, str(center_point), center_point, cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
                                motor_drive(center_point[0],center_point[1])

        #...loop    

        elif(mode==2):
            manual_mode_thread = Thread(target=manual_mode, args=(), kwargs={})    
            manual_mode_thread.start()
            manual_mode_thread.join()
        else:
            print("error")    
    


if __name__ == '__main__':
    main(**vars(parse_opt))
   

    '''
    수정중 근데 이러면 못빠져나오잖아.. 생각해보니까 무거운 부분이 프로그램 앞부분이라 리소스 많이 차지해서 뒷부분은 불가능할듯..
    '''
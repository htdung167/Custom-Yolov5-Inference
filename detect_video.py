import torch
import torch.nn as nn
import cv2 as cv
import os
import time
import argparse
from utils_.general import drop_image

from model.yolov5_ import YOLOv5
from configparser import ConfigParser
import json
config = ConfigParser()
config.read("./config/yolov5.cfg")

DEVICE = str(config["model"]["DEVICE"])
DET_MODEL_TYPE = str(config["model"]["DET_MODEL_TYPE"])
DET_WEIGHT_PATH = str(config["model"]["DET_WEIGHT_PATH"])
DET_INPUT_SIZE = json.loads(config["model"]["DET_INPUT_SIZE"])
DET_CONF_THRES = float(config["model"]["DET_CONF_THRES"])
DET_IOU_THRES = float(config["model"]["DET_IOU_THRES"])
DET_MAX_DETECT = int(config["model"]["DET_MAX_DETECT"])

def run(
        source,
        show_result=False,
        save_video=False,
        name='exp'
):
    #--Create save folder
    if save_video:
        stt = 0
        save_path = os.path.join("experiments", name)
        while os.path.exists(save_path):
            stt += 1
            name_ = name + str(stt)
            save_path = os.path.join("experiments", name_)
        os.mkdir(save_path)

    #--Initialize model
    model = YOLOv5(
        weights_path=DET_WEIGHT_PATH,
        device=DEVICE,
        input_size=DET_INPUT_SIZE,
        conf_thres=DET_CONF_THRES,
        iou_thres=DET_IOU_THRES,
        max_detect=DET_MAX_DETECT
        )
    
    #--Create video writer
    cap = cv.VideoCapture(source)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if save_video:
        result = cv.VideoWriter(os.path.join(save_path, os.path.basename(source)), 
                            cv.VideoWriter_fourcc(*'MP4V'), 
                            fps, 
                            (frame_width, frame_height))
    # Loop
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #--Predict and show result
            det_pred, det_img = model(frame, show=show_result)  #x1y1x2y2
            #--Show video
            if show_result:
                cv.imshow('image',det_img)
                cv.waitKey(1)
            #--Save video
            if save_video:
                result.write(det_img)
            #--Print
            print(f">> Frame {frame_id}")
            frame_id += 1
        else:
            break
    cap.release()
    result.release()
            
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="data/1.mp4", help='path to a video')
    parser.add_argument('--show_result', action='store_true', help='show video')
    parser.add_argument('--save_video', action='store_true', help='save result video')
    parser.add_argument('--name', type=str, default="exp", help='save result to exp folder')
    opt = parser.parse_args()
    return opt

def main(opt):
    print(vars(opt))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
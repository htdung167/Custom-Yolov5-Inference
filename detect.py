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
        big_source,
        show_result=False,
        save_txt=False,
        save_image=False,
        name='exp'
):
    #--Create save folder
    if save_txt or save_image:
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

    for source in os.listdir(big_source):
        #--Create sub folder
        save_img_path = os.path.join(save_path, source)
        os.mkdir(save_img_path)
        #--List image
        if os.path.isdir(os.path.join(big_source, source)):
            source = [os.path.join(big_source, source, x) for x in os.listdir(os.path.join(big_source, source))]
        else:
            raise TypeError("Struture are not allowed!")
        len_source = len(source)

        for idx, img_path in enumerate(source):
            #--Preprocess
            img_org = cv.imread(img_path)

            #--Predict and show result
            det_pred, det_img = model(img_org, show=show_result)  #x1y1x2y2
            
            # Cut detected object
            det_pred = det_pred[0].cpu().detach().numpy()
            bboxes = det_pred[:, 0:4]
            lst_drop_img = drop_image(img_org, bboxes)
            if show_result:
                for dropped_img in lst_drop_img:
                    cv.imshow("", dropped_img)
                    cv.waitKey(1)

            print(f"{idx + 1}/{len_source}: {img_path} \n{det_pred}")

            #--Save result
            img_file = img_path.rsplit("/", 1)[-1].rsplit("\\", -1)[-1]
            if save_image:
                img_file = img_path.rsplit("/", 1)[-1].rsplit("\\", -1)[-1]
                if len(lst_drop_img) > 0:
                    cv.imwrite(os.path.join(save_img_path, img_file), img_org)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--big_source', type=str, default="data/1.jpg", help='path to a image or image folder')
    parser.add_argument('--show_result', action='store_true', help='show image and labels')
    parser.add_argument('--save_image', action='store_true', help='save dropped image results to *.jpg')
    parser.add_argument('--name', type=str, default="exp", help='save result to exp folder')
    opt = parser.parse_args()
    return opt

def main(opt):
    print(vars(opt))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
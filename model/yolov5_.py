
import numpy as np
import torch
import torch.nn as nn
import sys
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import cv2 as cv
from utils_.plot import plot_bbox_label

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.augmentations import letterbox

class YOLOv5(nn.Module):
    def __init__(
        self,
        weights_path,
        device,
        input_size,
        conf_thres,
        iou_thres,
        max_detect,
        ):
        super(YOLOv5, self).__init__()
        #--Initialize
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_detect = max_detect
        self.classes = None

        #--Load model
        self.model = DetectMultiBackend(self.weights_path, device=self.device)
        self.stride, self.pt, self.names = self.model.stride, self.model.pt, self.model.names
        self.input_size = check_img_size(self.input_size, s=self.stride)
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.xyxy2xywh = xyxy2xywh
        self.letterbox = letterbox

        #--Plot
        self.list_label = list(self.names.values())
        self.list_color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(self.list_label))]
    
    def get_model(self):
        return self.model

    @torch.no_grad()
    def forward(self, image, show=False, show_conf=True):
        """Predict with Yolov5

        Parameters
        ----------
        image : np.ndarray (H, W, C)
            The BGR image read from opencv library.

        Returns
        -------
        result: tensor
            Result of yolov5. 
        """
        im = image.copy()
        # Reshape and pad image
        im = self.letterbox(im, self.input_size, stride=self.stride, auto=self.pt)[0]
        # print(self.stride) #32
        # print(self.pt) #True
        # HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))[::-1] 
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255.0
        #--Expand for batch dim
        if len(im.shape) == 3:
            im = im[None] 

        #--Inference
        pred = self.model(im) # ...,xywh

        #--Apply NMS
        pred = self.non_max_suppression(
            prediction=pred, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres, 
            max_det=self.max_detect) # [xyxy, conf, cls]

        #--Process detections
        for i, det in enumerate(pred):
            im0 = image.copy()
            if det is not None and len(det):
                # Rescale boxes from im to im0 size
                det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                pred[i][:, :4] = det[:, :4]

                #--Visualize
                for d in det:
                    bboxes = d[0:4]
                    klass = self.names[int(d[5])]
                    conf = round(float(d[4].numpy()), 2)
                    if show_conf:
                        im0 = plot_bbox_label(im0, bboxes, label=str(klass) + " " + str(conf), list_label=self.list_label, list_color=self.list_color)
                    else:
                        im0 = plot_bbox_label(im0, bboxes, label=str(klass), list_label=self.list_label, list_color=self.list_color)

            if show:
                cv.imshow("image", im0)
                cv.waitKey(1)
        return pred, im0
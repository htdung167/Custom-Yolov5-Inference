import cv2 as cv
import numpy as np

def drop_image(image, bboxes):
    """Drop image with bounding boxes.
    Parameters
    ---------
    images: np.ndarray
        BGR
    bboxes: np.ndarray
        <xyxy>

    Returns
    ---------
    lst_img: list
        List drop images
    """
    lst_img = []
    h_img, w_img = image.shape[:2]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = x1 - 1 if x1 > 0 else x1
        y1 = y1 - 1 if y1 > 0 else y1
        x2 = x2 + 1 if x2 < w_img else x2
        y2 = y2 + 1 if y2 < h_img else y2
        img = image[y1:y2, x1:x2]
        lst_img.append(img)
    return lst_img

def xyxy2normalizedxywh(img_shape, xyxy):
    h_img, w_img = img_shape[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    x_cen = (x1 + x2) / 2
    y_cen = (y1 + y2) / 2
    
    x_cen = round(x_cen / w_img, 6)
    y_cen = round(y_cen / h_img, 6)
    w = round(w / w_img, 6)
    h = round(h / h_img, 6)
    return x_cen, y_cen, w, h

def xyxy2xywh(img_shape, xyxy):
    h_img, w_img = img_shape[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    x_cen = int((x1 + x2) / 2)
    y_cen = int((y1 + y2) / 2)
    return x_cen, y_cen, w, h

def read_label_file(filee):
    result = []
    with open(filee, "r", encoding='utf-8') as f:
        content = f.read().split("\n")
    for line in content:
        if len(line) == 0:
            continue
        info = line.strip().split(" ")
        if len(info)==0:
            continue
        elif len(info) == 4:
            info = ["Empty",] + info
        content_LP, x, y, w, h = info
        x, y, w, h = float(x), float(y), float(w), float(h)
        result.append([content_LP, x, y, w, h])
    return result

def iou(image, bbox1, bbox2):
    """bbox: <x_normalized, y_normalized, w_normalized, h_normalized>"""
    h_img, w_img = image.shape[:2]
    
    # bbox1
    x, y, w, h = bbox1
    x, y, w, h = x * w_img, y*h_img, w*w_img, h*h_img
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    boxA = [x1, y1, x2, y2]

    # bbox2
    x, y, w, h = bbox2
    x, y, w, h = x * w_img, y*h_img, w*w_img, h*h_img
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    boxB = [x1, y1, x2, y2]

    # Iou
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou






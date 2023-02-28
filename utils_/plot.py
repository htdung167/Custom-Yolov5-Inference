import cv2 as cv

def plot_bbox_label(img, bbox, label):
    """bbox: <tl_x tl_y br_x br_y>"""
    im0 = img.copy()
    #--Draw detect result
    ##--Bounding box
    tl_x, tl_y, br_x, br_y = bbox
    im0 = cv.rectangle(
        img=im0, 
        pt1=(int(tl_x), int(tl_y)), 
        pt2=(int(br_x), int(br_y)), 
        color=(0, 0, 255), 
        thickness=1)
    ##--Text
    label_text = label
    text_size = cv.getTextSize(text=label_text, 
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        thickness=1)
    im0 = cv.rectangle(
        img=im0, 
        pt1=(int(tl_x), int(tl_y) - text_size[0][1] - 3), 
        pt2=(int(tl_x) + text_size[0][0], int(tl_y)), 
        color=(0, 0, 255), 
        thickness=-1,
        lineType=cv.LINE_AA)
    cv.putText(
        img=im0, 
        text=label_text, 
        org=(int(tl_x), int(tl_y)-3), 
        fontFace=cv.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.5, 
        color=(255, 255, 255), 
        thickness=1,
        lineType=cv.LINE_4)
    return im0
    

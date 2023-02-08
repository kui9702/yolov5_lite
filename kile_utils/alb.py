import albumentations as A
import cv2
import time

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def bBoxSafeRandomCrop(image, bboxes, category_ids, p=1):
    if len(category_ids.shape) != 1:
        category_ids = category_ids.reshape(-1)
    transform = A.Compose(
        [A.BBoxSafeRandomCrop(p=p)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )
    try:
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    except:
        # print("dasdadsadasdasdadada\n")
        return image, bboxes, category_ids
        
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2, type="yolo", xyxy="xyxy"):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    if type=="yolo":
        x_min, x_max, y_min, y_max = int((x_min-w/2)*img.shape[1]), int(((x_min + w/2))*img.shape[1]), int((y_min-h/2)*img.shape[0]), int((y_min + h/2)*img.shape[0])
    else:
        if xyxy != "xyxy":
            x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        else:
            x_min, x_max, y_min, y_max = int(x_min), int(w), int(y_min), int(h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img
 
def visualize(image, bboxes, category_ids, type="yolo", xyxy="xyxy"):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = str(category_id)
        img = visualize_bbox(img, bbox, class_name, type=type, xyxy=xyxy)
    cv2.imwrite(f"./{time.time()}.jpg", img)
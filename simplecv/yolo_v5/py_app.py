import cv2 as cv
import torch
from yolo.models.experimental import attempt_load
from yolo.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh
from yolo.utils.torch_utils import select_device, load_classifier


def load_model(weights, device, half, imgsz):
    # Initialize
    device = select_device(device)
    # half precision only supported on CUDA
    half = half if device.type != "cpu" else False

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    return model, device


classify = False
device = None
model = None
half = None
opt = None
modelc = None
names = None


def inference(img, device, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()

    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

# --- LEGACY/MISC FILE ---
# This script does not belong to any module in the target repository
# architecture (Computer Vision / Photonic Simulations / Deep Learning).
# It is kept here unlinked, for reference only, and is not imported by
# anything in src/ or apps/. See legacy_misc/README.md for details.
# ------------------------

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import GPT2Tokenizer, GPT2Model  # NLP part
from utils.general import scale_coords, non_max_suppression, xyxy2xywh
from utils.plots import Annotator, colors
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, time_sync

# Load NLP model for sequence analysis
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
nlp_model = GPT2Model.from_pretrained("gpt2")


@torch.no_grad()
def run(weights='yolov5s.pt', source='data/images', imgsz=640, conf_thres=0.25, iou_thres=0.45, device='',
        max_det=1000):
    # Initialize
    device = select_device(device)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)  # Load YOLOv5 model

    # Load data
    dataset = LoadImages(source, img_size=imgsz, stride=model.stride.max())

    # For tracking activity sequences
    activity_logs = []

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=3)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Log activity (bounding box positions and class)
                activity_sequence = []
                for *xyxy, conf, cls in det:
                    # Generate activity sequence (e.g., "car moving", "car stopped")
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(im0.shape)[[1, 0, 1, 0]]).view(
                        -1).tolist()
                    activity_sequence.append(f"class {int(cls)} at {xywh}")

                    # Annotate and display the bounding boxes on image
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                # Analyze sequence using NLP model
                sequence_text = " ".join(activity_sequence)  # Treat as a text sequence
                inputs = tokenizer(sequence_text, return_tensors="pt")
                outputs = nlp_model(**inputs)
                hidden_states = outputs.last_hidden_state  # use hidden states for further pattern detection
                activity_logs.append((path, activity_sequence))

            # Save the annotated image
            save_path = Path('runs/detect') / Path(path).name
            cv2.imwrite(str(save_path), annotator.result())

    # At the end, process the activity logs for final pattern analysis
    analyze_activity_patterns(activity_logs)


def analyze_activity_patterns(activity_logs):
    """
    Analyzes the collected activity logs for patterns such as abnormal driving or traffic behavior.
    """
    print("Analyzing activity patterns...")
    for log in activity_logs:
        path, sequence = log
        print(f"Pattern for {path}:")
        print(sequence)
        # Here you can add logic to detect patterns like sudden stops, lane changes, etc.


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

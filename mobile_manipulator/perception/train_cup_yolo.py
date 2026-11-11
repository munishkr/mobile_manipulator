#!/opt/yolo_venv/bin/python3
"""Fine-tune YOLOv8n on the auto-labelled cup dataset (collect_cup_dataset.py).

    /opt/yolo_venv/bin/python3 train_cup_yolo.py                 # 60 epochs, GPU 0
    /opt/yolo_venv/bin/python3 train_cup_yolo.py --epochs 5      # smoke test

Writes perception/runs/<name>/ (curves, metrics) and copies the best weights to
perception/models/cup_yolov8n.pt, which cup_detector.py loads by default.
Sized for the 4 GB RTX 500: yolov8n, 640 px, batch 16.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'datasets' / 'cup' / 'cup.yaml')
    ap.add_argument('--base', default='yolov8n.pt', help='pretrained COCO weights to start from')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--name', default='cup')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=16,
                    help='CPU torch threads; more oversubscribes the CPU and starves Gazebo')
    ap.add_argument('--out', type=Path, default=HERE / 'models' / 'cup_yolov8n.pt')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Training on {torch.cuda.get_device_name(0) if device == 0 else "CPU"}')
    (HERE / 'models').mkdir(exist_ok=True)
    model = YOLO(str(HERE / 'models' / args.base) if (HERE / 'models' / args.base).exists()
                 else args.base)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, project=str(HERE / 'runs'), name=args.name, exist_ok=True,
        patience=15, workers=args.workers, seed=0, plots=True,
        # The scene is synthetic and nearly constant: lean on colour/scale jitter so
        # the detector does not just memorise one lighting setup.
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=5.0, scale=0.4, translate=0.15,
        fliplr=0.5, mosaic=1.0, close_mosaic=10,
    )
    best = HERE / 'runs' / args.name / 'weights' / 'best.pt'
    shutil.copy2(best, args.out)
    metrics = YOLO(str(args.out)).val(data=str(args.data), device=device, plots=False)
    print(f'Best weights -> {args.out}')
    print(f'val mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  '
          f'precision={metrics.box.mp:.3f}  recall={metrics.box.mr:.3f}')


if __name__ == '__main__':
    main()

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolo11n.pt')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=100,
                batch=32,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=True,
                patience=10
                )

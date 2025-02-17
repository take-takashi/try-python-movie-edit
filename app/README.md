# app

## setup command

```bash
% poetry -V
Poetry (version 2.0.1)
% poetry init
% pyenv local 3.13.1
% poetry env use $(pyenv which python)
% poetry env info | grep -A 5 "Virtualenv" | grep "Python:" | awk '{print $2}'
3.13.1
% poetry shell
% deactivate
```

## poetry add

```bash
% poetry add -D py2app
% poetry add ultralytics opencv-python torch torchvision ffmpeg-python "numpy<=2.1.1"
% poetry add -D pyinstaller
```

## memo

- [ultralytics/ultralytics: Ultralytics YOLO11 🚀](https://github.com/ultralytics/ultralytics)  
  YOLOのモデルDL
- [akanametov/yolo-face: YOLO Face 🚀 in PyTorch](https://github.com/akanametov/yolo-face?tab=readme-ov-file)  
  YOLOのモデルDL（顔検出）

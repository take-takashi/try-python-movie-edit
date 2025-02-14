# try06

方針

- ✅顔検出ではなく、全身を検出する
- モザイクはガウスを採用
- モザイクは最大4,5フレーム保持したい
- 2フレームごとに検出
- 2フレーム目は同じ箇所にモザイク
- （範囲を少しだけ拡大？）
- マルチスレッドを完全にやめれていない？
- もはや顔検出ではなく、人物全体にする
- ✅python3.13ではどうだ？
- yolov11s.ptではどうだ？

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
% poetry add -D ipykernel
% poetry add ultralytics opencv-python torch torchvision "numpy<=2.1.1"
```

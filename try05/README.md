# try05

## setup commands

```bash
% poetry -V
Poetry (version 2.0.1)

# 今回はpython3.11.11を利用
% poetry init
% pyenv local 3.11.11
% poetry env use $(pyenv which python)
% poetry env info
% poetry shell
```

## poetry add

```bash
% poetry add -D ipykernel
% poetry add ultralytics opencv-python torch torchvision "numpy<=2.1.1"
% poetry add omegaconf
```

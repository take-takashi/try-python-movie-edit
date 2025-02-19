import sys
import os
import matplotlib

# キャッシュフォルダの変更
os.environ["MPLCONFIGDIR"] = "./mpl_cache"

print("cache: " + matplotlib.get_cachedir())

# ultralyticsをインポートする前にmatplotlib.font_managerがロードされているか確認
print("Before YOLO import:")
print("matplotlib.font_manager" in sys.modules)

# YOLOをインポート
from ultralytics import YOLO

# ultralyticsインポート後にmatplotlib.font_managerがロードされているか確認
print("After YOLO import:")
print("matplotlib.font_manager" in sys.modules)

# model = YOLO("yolov11n-face.pt")
# results = model("image.jpg", show=False)  # show=False にして可視化を防ぐ
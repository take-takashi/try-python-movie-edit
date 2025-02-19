import os
import sys
from pathlib import Path
import matplotlib

# キャッシュフォルダの変更
# （残念ながらultralyticsのimport前に実施する必要がある）
os.environ["MPLCONFIGDIR"] = "./.mpl_cache"
print(f"matplotlib: {matplotlib.get_cachedir()}")

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from queue import Queue
import threading
import ffmpeg
from com_func import resource_path

# -------------------------------------------------------------
# 1) 設定: デバイス・モデル・stride32リサイズ関数
# -------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_path = resource_path("assets/yolov11n-face.pt")
model = YOLO(model_path).to(device)

cv2.setNumThreads(cv2.getNumberOfCPUs())

def resize_to_stride32(image):
    """
    画像を YOLO のstride=32 の倍数 (height, width) にリサイズする
    """
    height, width = image.shape[:2]
    new_height = (height // 32) * 32 + (32 if height % 32 != 0 else 0)
    new_width  = (width  // 32) * 32 + (32 if width  % 32 != 0 else 0)
    if new_width == width and new_height == height:
        return image, width, height
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized, new_width, new_height

def blur_face(image, ksize=(15, 15)):
    """
    顔部分を縮小してガウシアンぼかしをかけてから元サイズに戻す
    """
    if image.size == 0:
        return image
    small = cv2.resize(image, ksize, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    return cv2.resize(blurred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

# -------------------------------------------------------------
# 2) 顔の位置補完 (IOU管理)
# -------------------------------------------------------------
def compute_iou(boxA, boxB):
    """
    2つのバウンディングボックス(boxA, boxB)に対するIoU(Intersection over Union)を計算
    box = (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def update_face_tracks(face_tracks, new_dets, iou_thresh=0.5, max_life=4):
    """
    前フレームまでの顔領域リスト(face_tracks)に、今フレームの検出結果(new_dets)を反映して更新する。
      - face_tracks: [(x1, y1, x2, y2, life), ... ]
      - new_dets:    [(x1, y1, x2, y2), ... ]
    iou_thresh: IoU がこの値以上なら同じ顔とみなす
    max_life:   ライフ上限(継続保持するフレーム数)
    """
    updated_tracks = []
    for (tx1, ty1, tx2, ty2, life) in face_tracks:
        updated_tracks.append([tx1, ty1, tx2, ty2, life - 1])
    for (nx1, ny1, nx2, ny2) in new_dets:
        best_iou = 0
        best_index = -1
        for i, (tx1, ty1, tx2, ty2, life) in enumerate(updated_tracks):
            iou_val = compute_iou((tx1, ty1, tx2, ty2), (nx1, ny1, nx2, ny2))
            if iou_val > best_iou:
                best_iou = iou_val
                best_index = i
        if best_iou >= iou_thresh and best_index >= 0:
            updated_tracks[best_index][0] = nx1
            updated_tracks[best_index][1] = ny1
            updated_tracks[best_index][2] = nx2
            updated_tracks[best_index][3] = ny2
            updated_tracks[best_index][4] = max_life
        else:
            updated_tracks.append([nx1, ny1, nx2, ny2, max_life])
    filtered_tracks = []
    for t in updated_tracks:
        if t[4] > 0:
            filtered_tracks.append(t)
    return filtered_tracks

# -------------------------------------------------------------
# 3) 動画書き出しを並列化するクラス
# -------------------------------------------------------------
class FrameWriter(threading.Thread):
    """
    別スレッドでフレームを書き込む。
      - メインスレッドでフレームを queue に put する
      - ここで queue.get() して VideoWriter.write() する
    """
    def __init__(self, video_writer, frame_queue):
        super().__init__()
        self.video_writer = video_writer
        self.frame_queue = frame_queue
        self.stop_signal = False

    def run(self):
        while True:
            if self.stop_signal and self.frame_queue.empty():
                break
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except:
                continue
            self.video_writer.write(frame)
            self.frame_queue.task_done()

    def stop(self):
        self.stop_signal = True

# -------------------------------------------------------------
# 4) メイン処理: バッチ推論 + 2フレームに1回検出 + IOU補完 + 並列書き出し
# -------------------------------------------------------------
def process_video(input_path, output_path, batch_size=4):
    # ffmpeg.probe で動画のメタデータを取得し、回転情報を抽出
    metadata = ffmpeg.probe(input_path)
    stream0 = metadata['streams'][0]
    rotate_tag = stream0.get('tags', {}).get('rotate', None)
    if rotate_tag is None and 'side_data_list' in stream0:
        for side in stream0['side_data_list']:
            if side.get('side_data_type') == 'Display Matrix' and 'rotation' in side:
                rotate_tag = side['rotation']
                break
    if rotate_tag is None:
        rotate_tag = '0'
    rotation_angle = int(rotate_tag)
    print(f"動画の回転情報: {rotation_angle}°")

    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 回転補正後の出力サイズを設定（90°または270°なら幅と高さを入れ替え）
    if rotation_angle in [90, 270]:
        output_size = (height, width)
    else:
        output_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    frame_queue = Queue(maxsize=10)
    writer_thread = FrameWriter(out_writer, frame_queue)
    writer_thread.start()

    detect_interval = 2
    frame_count = 0
    start_time = time.time()

    face_tracks = []
    max_life = 4

    frame_batch = []
    original_frames = []
    detect_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----- 回転補正: メタデータの回転情報に基づいて各フレームを補正 -----
        # 補正角度を正の値に変換
        norm_angle = abs(rotation_angle)
        if norm_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif norm_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif norm_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # --------------------------------------------------------------------

        frame_count += 1

        resized_frame, new_w, new_h = resize_to_stride32(frame)
        frame_batch.append(resized_frame)
        original_frames.append(frame)
        detect_flags.append(frame_count == 1 or (frame_count % detect_interval == 0))

        if len(frame_batch) == batch_size or frame_count == total_frames:
            sub_tensors = []
            sub_indices = []
            for i, (f, flag) in enumerate(zip(frame_batch, detect_flags)):
                if flag:
                    tensor = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
                    sub_tensors.append(tensor)
                    sub_indices.append(i)
            if len(sub_tensors) > 0:
                batch_tensor = torch.cat(sub_tensors, dim=0)
                results = model.predict(
                    batch_tensor, 
                    verbose=False,
                    imgsz=(new_w, new_h),
                    conf=0.25,
                    iou=0.3,
                    agnostic_nms=True
                )
            else:
                results = []

            detection_results = [[] for _ in range(len(frame_batch))]
            for r_i, r in enumerate(results):
                i_batch_index = sub_indices[r_i]
                new_faces = []
                for box in r.boxes.xyxy:
                    x1_r, y1_r, x2_r, y2_r = map(int, box)
                    x1 = int(x1_r * width / new_w)
                    y1 = int(y1_r * height / new_h)
                    x2 = int(x2_r * width / new_w)
                    y2 = int(y2_r * height / new_h)
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        continue
                    new_faces.append((x1, y1, x2, y2))
                detection_results[i_batch_index] = new_faces

            for i in range(len(frame_batch)):
                new_dets = detection_results[i] if i < len(detection_results) else []
                face_tracks = update_face_tracks(face_tracks, new_dets, iou_thresh=0.5, max_life=max_life)
                for (fx1, fy1, fx2, fy2, _) in face_tracks:
                    face_roi = original_frames[i][fy1:fy2, fx1:fx2]
                    original_frames[i][fy1:fy2, fx1:fx2] = blur_face(face_roi)
                frame_queue.put(original_frames[i])

            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"Frame {frame_count}/{total_frames}, Estimated remaining: {remaining:.1f} sec")
            frame_batch = []
            original_frames = []
            detect_flags = []

    cap.release()
    writer_thread.stop()
    writer_thread.join()
    out_writer.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.1f} seconds")

# -------------------------------------------------------------
# 5) 実行例
# -------------------------------------------------------------
if __name__ == "__main__":

    # 引数チェック
    if len(sys.argv) < 3:
        print("使い方: python app.py 入力動画ファイル 出力動画ファイル")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    path = Path(output_file)
    output_noaudio_file = path.with_name(f"{path.stem}_noaudio{path.suffix}")

    print("入力ファイル:", input_file)
    print("中間ファイル:", output_noaudio_file)
    print("出力ファイル", output_file)

    process_video(input_file, output_noaudio_file, batch_size=4)

    # 音声、メタデータの合成
    video = ffmpeg.input(output_noaudio_file).video  # 出力済み映像
    audio = ffmpeg.input(input_file).audio   # 元の音声（存在すれば）

    # ffmpeg の出力設定:
    # - map_metadata=1 で、2 番目の入力（input.mp4）のメタデータをコピー
    # - vcodec='copy', acodec='copy' で再エンコードせずコピーする

    ffmpeg.output(video, audio, output_file,
                map_metadata=1,
                vcodec='copy',
                acodec='copy').run(overwrite_output=True, quiet=True)

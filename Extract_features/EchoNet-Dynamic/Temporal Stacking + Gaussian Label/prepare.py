import os
import cv2
import pandas as pd
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-videos', type=int, default=200)
args = parser.parse_args()

#MAX_VIDEOS = args.max_videos
MAX_VIDEOS = 200

# Tạo thư mục chứa dữ liệu 2.5D
SAVE_DIR = 'frames_25d'
os.makedirs(SAVE_DIR, exist_ok=True)

print("Đọc file VolumeTracings.csv...")
df = pd.read_csv("../../../data/pediatric_echo_avi/A4C/VolumeTracings.csv")

# Chuyển giá trị frame sang số nguyên để tránh lỗi khi tính khoảng cách
df['Frame'] = pd.to_numeric(df['Frame'], errors='coerce').astype('Int64')
target_frames_dict = df.groupby('FileName')['Frame'].apply(lambda s: [int(v) for v in s.dropna()]).to_dict()

video_dir = '../../../data/pediatric_echo_avi/A4C/Videos/'
sigma = 2.0 
labels_data = []
processed_count = 0

print("Bắt đầu trích xuất videos")

for video_name, target_frames in target_frames_dict.items():
    if processed_count >= MAX_VIDEOS:
        break
        
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video {video_name}")
        continue
        
    # 1. Đọc toàn bộ khung hình của 1 video và chuyển sang ảnh Xám (Grayscale)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        
    cap.release()

    if len(frames) < 3:
        continue

    # 2. Kẹp 3 frames liên tiếp (Từ frame số 1 đến frame áp chót)
    for i in range(1, len(frames) - 1):
        # Trộn [t-1, t, t+1] thành 3 kênh của một ảnh màu
        stacked_img = cv2.merge([frames[i-1], frames[i], frames[i+1]])
        
        # 3. Tính điểm Gaussian (0.0 đến 1.0)
        min_distance = min([abs(i - tf) for tf in target_frames])
        score = np.exp(-(min_distance**2) / (2 * sigma**2))
        
        # Lọc dữ liệu: Giữ lại những frame có điểm > 0.01 và chỉ lấy ngẫu nhiên 10% các frame rác/nền (điểm ~ 0)
        if score > 0.01 or random.random() < 0.1:
            img_name = f"{video_name.split('.')[0]}_f{i}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, img_name), stacked_img)
            
            labels_data.append({
                'filename': img_name,
                'score': score
            })
            
    processed_count += 1
    print(f"Đã xử lý {processed_count}/{MAX_VIDEOS}: {video_name}")

# Lưu bảng điểm
pd.DataFrame(labels_data).to_csv('labels_25d.csv', index=False)
print(f"\nHoàn tất! Dữ liệu đã lưu tại thư mục '{SAVE_DIR}' và file 'labels_25d.csv'.")
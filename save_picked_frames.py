import os
import cv2
import h5py
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RESULT_FILE = 'output_hybrid/result.h5'

# Thư mục chứa video gốc
VIDEO_DIR = 'data/pediatric_echo_avi/A4C/Videos/'

# Thư mục đích để lưu ảnh
SAVE_DIR = 'Set_of_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_and_save_frames():
    print(f"[*] Đang đọc kết quả từ: {RESULT_FILE}")
    
    # 1. Đọc file kết quả
    with h5py.File(RESULT_FILE, 'r') as f:
        # Lấy danh sách tên các video đã được test
        video_names = list(f.keys())
        
        for vid_name in video_names:
            machine_summary = f[vid_name]['machine_summary'][...] 
            # Chuyển mảng nhị phân (0, 1) thành mảng chứa index của các frame được chọn
            picked_indices = np.where(machine_summary == 1)[0]
            
            if len(picked_indices) == 0:
                continue
                
            # 2. Mở video siêu âm gốc tương ứng
            video_path = os.path.join(VIDEO_DIR, f"{vid_name}.avi")
            if not os.path.exists(video_path):
                print(f"[!] Không tìm thấy video gốc: {video_path}")
                continue
                
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            saved_count = 0
            
            # 3. Quét qua video và lưu các frame đc pick
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx in picked_indices:
                    img_name = f"{vid_name}_frame_{frame_idx}.jpg"
                    save_path = os.path.join(SAVE_DIR, img_name)
                    
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                    
                frame_idx += 1
                
            cap.release()
            print(f"[+] Video {vid_name}: Đã lưu {saved_count} frames vào {SAVE_DIR}/")

if __name__ == '__main__':
    extract_and_save_frames()
    print(f"[*] HOÀN TẤT! Toàn bộ ảnh đã nằm trong thư mục: {SAVE_DIR}")
#!/bin/bash

# ==========================================
# 1. DỌN DẸP MÔI TRƯỜNG CŨ
# ==========================================
rm -rf output_hybrid


# ==========================================
# 2. HUẤN LUYỆN MÔ HÌNH RL
# ==========================================
PYTHONPATH=. python3 scripts/script_us_vsumm_train_test.py \
    --gpu 1 \
    --max-epoch 50 \
    --save-dir output_hybrid \
    --reward-type Rall \
    --train-model sup \
    --lr 0.00001


# ==========================================
# 3. TỰ ĐỘNG DỰ ĐOÁN (INFERENCE)
# ==========================================
echo "Đang tìm file trọng số (.pth) tốt nhất"
# Lệnh này sẽ tự động mò vào thư mục và lấy cái file .pth mới nhất/đạt điểm cao nhất
BEST_MODEL=$(ls output_hybrid/sup/split0/*.pth | sort -t 'F' -k 2 -nr | head -n 1)

if [ -z "$BEST_MODEL" ]; then
    echo "Không tìm thấy file model .pth"
    exit 1
fi

echo "[*] Đang chạy đánh giá để trích xuất file kết quả H5"

# Chạy file evaluate
PYTHONPATH=. python3 scripts/evaluate.py \
    --model "$BEST_MODEL" \
    --save-name output_hybrid/result.h5


# ==========================================
# 4. LƯU KHUNG HÌNH (PICKED FRAMES)
# ==========================================
python3 save_picked_frames.py
# Fetal & Pediatric Echocardiogram Summarization using Deep Reinforcement Learning

> **Dự án Nghiên cứu:** Ứng dụng Deep Reinforcement Learning vào bài toán Tóm tắt Video Siêu âm tim (Medical Ultrasound Video Summarization) nhằm tự động trích xuất các khung hình chẩn đoán quan trọng (đỉnh Tâm thu/Tâm trương) hỗ trợ chẩn đoán bệnh lý tim bẩm sinh.

---

## Giới thiệu chung

Trong chẩn đoán siêu âm tim định lượng, việc xác định chính xác các khung hình ở cuối thì tâm trương (ED) và cuối thì tâm thu (ES) là bước tiên quyết. Tuy nhiên, thao tác thủ công tốn rất nhiều thời gian và phụ thuộc vào kinh nghiệm của bác sĩ.

Dự án này đề xuất một luồng máy học tự động toàn phần (End-to-End Pipeline) bao gồm 2 giai đoạn:

1. **Feature Extraction:** Áp dụng kỹ thuật Temporal Stacking (kẹp 3 frames liên tiếp tạo ảnh 2.5D) kết hợp gán nhãn Gaussian mềm. Fine-tune mạng ResNet18 để trích xuất các ma trận đặc trưng 512 chiều.
2. **Deep Reinforcement Learning (RL):** Xây dựng một Agent RL để duyệt qua chuỗi đặc trưng và quyết định giữ/bỏ các khung hình, đối mặt với bài toán phần thưởng thưa thớt (Sparse Reward) trong không gian siêu âm tĩnh.


## Folder Structure

```text
Fetal-congenital-heart-disease-Echonet/
│
├── data/                               # Chứa dữ liệu thô và file đặc trưng 
│   ├── pediatric_echo_avi/             # Video .avi và file VolumeTracings.csv gốc
│   ├── echonet_dynamic.h5           # Dữ liệu đặc trưng tập người lớn
│   └── echonet_pediatric_a4c.h5   # Dữ liệu đặc trưng tập trẻ em (Sau khi extract)
│
├── Extract_features/                   # Giai đoạn 1: Tiền xử lý & Trích xuất đặc trưng
│   └── EchoNet-Dynamic/
│       └── Temporal Stacking + Gaussian Label/
│           ├── prepare.py              # Cắt frame, kẹp ảnh 2.5D, tính điểm Gaussian
│           ├── train.py                # Fine-tune mạng ResNet18
│           └── extract.py              # Đưa qua ResNet18 để lấy vector đặc trưng lưu ra file .h5
│
├── scripts/                            # Giai đoạn 2: Khởi chạy mô hình Reinforcement Learning
│   └── script_us_vsumm_train_test.py   # Script huấn luyện chính cho mạng RL
│
├── train.sh                            # File thực thi bash chạy nhanh mô hình
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn

# Tạo và kích hoạt môi trường ảo
    python3 -m venv venv
    venv\Scripts\activate   # Đối với Windows

# Cài đặt thư viện
pip install torch torchvision numpy pandas opencv-python h5py tqdm

## Chuẩn bị Dữ liệu (Data Preparation)

Do kích thước dữ liệu y khoa rất lớn, toàn bộ tập dữ liệu và trọng số mạng (`.pth`) không được tải trực tiếp lên kho lưu trữ này.

1. Tải thư mục `data.zip` và trọng số ResNet18 tại đây: 
    // Dataset 1: https://echonet.github.io/dynamic/
    // Dataset 2: https://echonet.github.io/pediatric/
2. Giải nén và đặt thư mục `data/` vào thư mục gốc của dự án.


## Cách chạy

### Giai đoạn 1: Feature Extraction Pipeline
Di chuyển vào thư mục xử lý dữ liệu:
```bash
cd "Extract_features/EchoNet-Dynamic/Temporal Stacking + Gaussian Label"

B1: Tạo dữ liệu ảnh & Nhãn (Prepare):
Bash:
    python3 prepare.py
Đầu ra: Thư mục chứa ảnh 2.5D và file labels_25d.csv.

B2: Train:
Bash:
    python3 train.py
Đầu ra: File trọng số mô hình best_echonet_25d.pth.

B3: Trích xuất ra định dạng H5:
Bash
    python3 extract.py
Đầu ra: File .h5 tự động được đẩy ra thư mục data/.

### Giai đoạn 2: Huấn luyện mạng Reinforcement Learning
Quay trở lại thư mục gốc dự án và khởi chạy file Bash: 
    cd ../../../
    sh train.sh
(Để thay đổi tập dữ liệu huấn luyện, mở file script_us_vsumm_train_test.py và sửa biến BASE_DATA_PATH trỏ tới file .h5 tương ứng trong thư mục data/). 
Đầu ra: Thư mục Set_of_frames chứa các frames được chọn



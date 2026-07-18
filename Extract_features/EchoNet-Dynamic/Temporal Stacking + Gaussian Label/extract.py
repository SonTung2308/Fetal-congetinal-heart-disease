import os
import cv2
import torch
import h5py
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# --- KHỞI TẠO MODEL ---
class EchoNet25DModel(nn.Module):
    def __init__(self):
        super(EchoNet25DModel, self).__init__()
        self.base_model = models.resnet18(weights=None)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.base_model(x)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị tính toán: {device}")

    # Load tạ
    model_weight = 'best_echonet_25d.pth'
    model = EchoNet25DModel()
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model = model.to(device)
    model.eval()

    # Hook lấy 512-D
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.view(output.size(0), -1)
        return hook
    model.base_model.avgpool.register_forward_hook(get_activation('features'))

    # Transform (Y hệt validation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("[*] Đọc thông tin từ file CSV...")
    df = pd.read_csv('labels_25d.csv')
    df['video_name'] = df['filename'].apply(lambda x: "_".join(x.split('_')[:-1]))
    df['frame_idx'] = df['filename'].apply(lambda x: int(''.join(filter(str.isdigit, x.split('_')[-1]))))
    unique_videos = df['video_name'].unique()
    
    output_h5 = '../../../data/echonet_pediatric_a4c.h5'
    batch_size = 32

    with h5py.File(output_h5, 'w') as h5_file:
        for vid_name in tqdm(unique_videos, desc='Tiến độ'):
            # Lọc ảnh của video này và sắp xếp đúng thứ tự thời gian
            vid_df = df[df['video_name'] == vid_name].sort_values('frame_idx')
            
            tensor_list = []
            labels_list = []
            
            # Đọc từng ảnh từ ổ cứng
            for _, row in vid_df.iterrows():
                img_path = os.path.join('frames_25d', row['filename'])
                img = cv2.imread(img_path)
                
                # CV2 đọc mặc định là BGR, chuyển về RGB để giống PIL Image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                tensor_list.append(val_transform(img))
                labels_list.append(row['score'])
                
            video_tensor = torch.stack(tensor_list).to(device)
            labels = np.array(labels_list, dtype=np.float32)
            extracted_features = []

            # Đưa qua ResNet
            with torch.no_grad():
                for i in range(0, len(video_tensor), batch_size):
                    batch = video_tensor[i:i + batch_size]
                    _ = model(batch)
                    extracted_features.append(activation['features'].cpu().numpy())

            final_features = np.concatenate(extracted_features, axis=0)

            # Lưu file H5
            grp = h5_file.create_group(vid_name.split('.')[0])
            grp.create_dataset('features', data=final_features, compression='gzip')
            grp.create_dataset('gtscore', data=labels, compression='gzip')
            grp.create_dataset('n_frames', data=final_features.shape[0])
            grp.create_dataset('picks', data=np.arange(final_features.shape[0], dtype=np.int32))

    print(f'\n[*] Xong! File H5 Nhãn Cứng đã sẵn sàng tại: {output_h5}')

if __name__ == '__main__':
    main()
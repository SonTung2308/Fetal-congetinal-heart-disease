import torch
import h5py
import os
import os.path as osp
import vsum_tools, eval_tools
from tabulate import tabulate
import numpy as np


def evaluate(args, model, dataset, proportion=0.15):

    # ==============================================================
    # [THÊM MỚI] 1. Mở sẵn một file H5 để hứng dữ liệu
    # ==============================================================
    result_path = osp.join(args.save_dir, 'result.h5')
    h5_res = h5py.File(result_path, 'w')
    
    with torch.no_grad():
        model.eval()
        meanFscores = []
        meanPrecision = []
        meanRecall = []

        if args.verbose: table = [["No.", "Video", "F-score"]]
        sum_proportions = [0.15, 0.25, 0.35, 0.45]
        
        for key_idx, key in enumerate(args.test_keys):

            # load gt_score from h5 file
            gtscore = dataset[key]['gtscore'][...]

            video_info = dataset[key]
            positions = video_info['picks'][...]

            seq = video_info['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)

            mean = seq.mean()
            std = seq.std()
            seq = (seq - mean) / (std + 1e-5)

            if torch.cuda.is_available():
                seq = seq.cuda()
                
            sig_probs = model(seq)
            probs_importance = sig_probs.data.cpu().squeeze().numpy()

            # generate gt summary on-fly according to different proportion constrains
            fscore_prop = []
            precision_prop = []
            recall_prop = []
            
            for p in range(len(sum_proportions)):
                prop = sum_proportions[p]
                # generate gt summary on-fly according to different propoption constrains
                gt_summary, _, _ = vsum_tools.generate_summary(gtscore, video_info, positions, proportion=prop)
                machine_summary, pred_probs_up, pick_segs \
                            = vsum_tools.generate_summary(probs_importance, video_info, positions, proportion=prop)
                f_score, precision, recall = eval_tools.evaluate_summary(machine_summary, gt_summary)
                
                fscore_prop.append(f_score)
                precision_prop.append(precision)
                recall_prop.append(recall)

                # ==============================================================
                # [THÊM MỚI] 2. Trích xuất mảng danh sách frame ra ổ cứng
                # (Chỉ lưu kết quả ở mức chuẩn p=0 tương đương tỉ lệ nén 15%)
                # ==============================================================
                if p == 0: 
                    v_data = h5_res.create_group(key)
                    v_data.create_dataset('machine_summary', data=machine_summary)
                    v_data.create_dataset('gtscore', data=gtscore)
                    v_data.create_dataset('probs_importance', data=probs_importance)
            
            meanFscores.append(fscore_prop)
            meanPrecision.append(precision_prop)
            meanRecall.append(recall_prop)

            if args.verbose:
                # Fix hiển thị f_score cho chuẩn xác với tỉ lệ 15%
                table.append([key_idx+1, key, "{:.1%}".format(fscore_prop[0])])

    if args.verbose:
        print(tabulate(table))

    # ==============================================================
    # [THÊM MỚI] 3. Đóng gói file và thông báo thành công
    # ==============================================================
    h5_res.close()
    print(f"[+] THÀNH CÔNG: Đã xuất danh sách frame chẩn đoán ra file: {result_path}")

    mean_fscores = np.mean(meanFscores, axis=0)
    mean_precision = np.mean(meanPrecision, axis=0)
    mean_recall = np.mean(meanRecall, axis=0)
    return mean_fscores, mean_precision, mean_recall

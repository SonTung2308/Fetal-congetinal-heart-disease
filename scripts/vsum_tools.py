import numpy as np
import h5py
import cv2
import torch
from knapsack import knapsack_dp
import math


def generate_summary(scores, video_info, positions, proportion=0.15):
   
    n_frames = video_info['n_frames'][()]
    limit = int(math.floor(n_frames * proportion))
    
    if limit == 0:
        limit = 1
        
    scores = np.array(scores).flatten()
    top_indices = np.argsort(scores)[-limit:]
    
    summary = np.zeros(n_frames, dtype=int)
    for idx in top_indices:
        if idx < n_frames:
            summary[idx] = 1
            
    return summary, np.array([]), np.array([])


def upsample_scores(pred_score, n_frames, positions):

    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    for idx in range(len(positions) - 1):
        pos_cur, pos_next = positions[idx], positions[idx + 1]

        if idx == len(pred_score):
            frame_scores[pos_cur:pos_next] = 0
        else:
            frame_scores[pos_cur:pos_next] = pred_score[idx]
    return frame_scores


def upsample_scores_interp(ypred, n_frames, positions):

    frame_scores_intep = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)

    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    for idx in range(len(positions) - 1):
        pos_cur, pos_next = positions[idx], positions[idx + 1]

        if idx > 0 and idx < len(ypred) - 1:  # perform linear interpolation of the prediction values
            for k in range(0, 16):
                frame_scores_intep[pos_cur + k] = ypred[idx - 1] + k * (ypred[idx] - ypred[idx - 1]) / 16.0
        elif idx == len(ypred):
            frame_scores_intep[pos_cur:-1] = ypred[idx - 1]
        #     for k in range(0, 16): frame_scores_intep[pos_cur+k] = ypred[idx-1] - k * ypred[idx-1]/16.0
        else:
            frame_scores_intep[pos_cur:pos_next] = ypred[idx]

    return frame_scores_intep


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break

    return oracle_summary

def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8);
        recall = overlap / (true_sum + 1e-8);
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)





import os
import torch
import scipy.io
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from torch.nn.functional import interpolate

# 경로 설정
types = ['depth', 'illumination', 'normal', 'reflectance']
pred_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/convnext_v2_attention/3000/mat/{}/mat'
gt_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/{}/'

# Thresholds
thresholds = torch.linspace(0.01, 0.99, 99, device='cuda')
maxDist = 0.0075  # maxDist 설정
target_size = (321, 481)
thin = 1  # thin 설정

# Define eps
eps = np.finfo(float).eps

def interp(I, x, y):
    h, w = I.shape
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    dx0 = x - x0.float()
    dy0 = y - y0.float()
    dx1 = 1 - dx0
    dy1 = 1 - dy0

    x0 = torch.clamp(x0, 0, w - 1)
    x1 = torch.clamp(x1, 0, w - 1)
    y0 = torch.clamp(y0, 0, h - 1)
    y1 = torch.clamp(y1, 0, h - 1)

    Ia = I[y0, x0]
    Ib = I[y1, x0]
    Ic = I[y0, x1]
    Id = I[y1, x1]

    return Ia * dx1 * dy1 + Ib * dx1 * dy0 + Ic * dx0 * dy1 + Id * dx0 * dy0

def edges_nms(E0, O, r, s, m):
    E = E0.clone()
    h, w = E0.shape
    cosO = torch.cos(O)
    sinO = torch.sin(O)

    for x in range(w):
        for y in range(h):
            e = E[y, x]
            if e == 0:
                continue
            e *= m
            for d in range(-r, r + 1):
                if d == 0:
                    continue
                x_d = x + d * cosO[y, x]
                y_d = y + d * sinO[y, x]
                if e < interp(E0, x_d, y_d):
                    E[y, x] = 0
                    break

    s = min(s, w // 2, h // 2)
    for x in range(s):
        for y in range(h):
            E[y, x] *= x / s
            E[y, w - 1 - x] *= x / s
    for x in range(w):
        for y in range(s):
            E[y, x] *= y / s
            E[h - 1 - y, x] *= y / s

    return E

def load_and_resize_mat_files(pred_dirs, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dirs[0]) if f.endswith('.mat')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.mat')])

    preds = []
    gts = []

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), desc="Loading and resizing files", total=len(pred_files)):
        pred_data = []
        for pred_dir in pred_dirs:
            pred_path = os.path.join(pred_dir, pred_file)
            pred = torch.tensor(scipy.io.loadmat(pred_path)['result'], device='cuda')
            if pred.shape != target_size:
                pred = interpolate(pred.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()
            pred_data.append(pred)

        gt_path = os.path.join(gt_dir, gt_file)
        gt = torch.tensor(scipy.io.loadmat(gt_path)['groundTruth'][0, 0]['Boundaries'][0, 0], device='cuda')
        if gt.shape != target_size:
            gt = interpolate(gt.unsqueeze(0).unsqueeze(0), size=target_size, mode='nearest').squeeze()

        # 비최대 억제 적용
        pred_data = torch.stack(pred_data, dim=-1)
        max_pred = torch.max(pred_data, dim=-1)[0]
        pred_data = (pred_data == max_pred.unsqueeze(-1)).float() * pred_data
        
        E0 = pred_data.squeeze(-1)

        gts.append(gt)
        preds.append(E0)

    return torch.stack(preds), torch.stack(gts)

def thin_image(E1):
    # 이진화된 이미지를 얇게 하는 함수
    device = E1.device
    E1_np = E1.cpu().numpy()
    from skimage.morphology import thin
    E1_thin = thin(E1_np).astype(np.float32)
    return torch.tensor(E1_thin, device=device)

def calculate_tp_fp_fn(preds, gts, thresholds, maxDist, image_diagonal, thin):
    tp_sum = torch.zeros(len(thresholds), device='cuda')
    fp_sum = torch.zeros(len(thresholds), device='cuda')
    fn_sum = torch.zeros(len(thresholds), device='cuda')
    tn_sum = torch.zeros(len(thresholds), device='cuda')

    for t_idx, threshold in enumerate(tqdm(thresholds, desc="Thresholds", leave=False)):
        for i in range(preds.shape[0]):
            E1 = (preds[i] >= max(eps, threshold)).float()
            if threshold > 0 and thin:
                E1 = thin_image(E1)
            
            gt_edge_pixels = torch.nonzero(gts[i], as_tuple=False).float()
            pred_edge_pixels = torch.nonzero(E1, as_tuple=False).float()

            if len(gt_edge_pixels) == 0:
                fp_sum[t_idx] += torch.sum(E1)
                tn_sum[t_idx] += torch.sum(1 - E1)
            else:
                if pred_edge_pixels.numel() == 0:
                    tp = torch.tensor(0.0, device='cuda')
                    fp = torch.tensor(0.0, device='cuda')
                    fn = len(gt_edge_pixels)
                else:
                    dist_matrix = torch.cdist(gt_edge_pixels, pred_edge_pixels)
                    if dist_matrix.numel() == 0:
                        tp = torch.tensor(0.0, device='cuda')
                        fp = len(pred_edge_pixels)
                        fn = len(gt_edge_pixels)
                    else:
                        tp = (dist_matrix.min(dim=0)[0] <= maxDist * image_diagonal[i]).sum()
                        fp = pred_edge_pixels.size(0) - tp
                        fn = (dist_matrix.min(dim=1)[0] > maxDist * image_diagonal[i]).sum()

                tn = torch.sum(1 - E1) - fn

                tp_sum[t_idx] += tp
                fp_sum[t_idx] += fp
                fn_sum[t_idx] += fn
                tn_sum[t_idx] += tn

    return tp_sum, fp_sum, fn_sum, tn_sum

def interpolate_ap(precision, recall):
    recall, recall_indices = np.unique(recall, return_index=True)
    precision = precision[recall_indices]

    # Interpolate precision values
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.interp(recall_interp, recall, precision)

    # Compute the average precision
    ap = np.mean(precision_interp)
    return ap

def main():
    pred_dirs = {t: [pred_dir_template.format(t)] for t in types}

    for t in types:
        gt_dir = gt_dir_template.format(t)

        preds, gts = load_and_resize_mat_files(pred_dirs[t], gt_dir)

        image_diagonal = torch.sqrt(torch.tensor(target_size[0]**2 + target_size[1]**2, device='cuda').float()).repeat(preds.shape[0])

        tp_sum, fp_sum, fn_sum, tn_sum = calculate_tp_fp_fn(preds, gts, thresholds, maxDist, image_diagonal, thin)

        # 정밀도, 재현율 및 ODS 계산
        precision = tp_sum / torch.where(tp_sum + fp_sum == 0, torch.tensor(1e-10, device='cuda'), tp_sum + fp_sum)
        recall = tp_sum / torch.where(tp_sum + fn_sum == 0, torch.tensor(1e-10, device='cuda'), tp_sum + fn_sum)
        f_score = 2 * precision * recall / torch.where(precision + recall == 0, torch.tensor(1e-10, device='cuda'), precision + recall)
        ods = f_score.max().item()

        # Tensor를 numpy 배열로 변환하여 sklearn 함수



        # Tensor를 numpy 배열로 변환하여 sklearn 함수 사용
        precision_np = precision.cpu().numpy()
        recall_np = recall.cpu().numpy()

        # AP를 보간하여 계산
        average_precision = interpolate_ap(precision_np, recall_np)

        print(f"Average Precision for {t}: {average_precision:.4f}")
        print(f"ODS for {t}: {ods:.4f}")

if __name__ == "__main__":
    main()



       




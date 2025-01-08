

import os
import torch
import scipy.io
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from torch.nn.functional import interpolate
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import math

# 경로 설정
types = ['depth', 'illumination', 'normal', 'reflectance']
pred_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/convnext_v2_case10_full_bathcnorm/2000/mat/{}/mat'
gt_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/{}/'

# Thresholds
thresholds = torch.linspace(0.01, 0.99, 99, device='cuda')
maxDist = 0.0075  # maxDist 설정
target_size = (321, 481)
thin = 1  # thin 설정

# Define eps
eps = np.finfo(float).eps

class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

def k_of_n(k, n):
    return np.random.choice(n, k, replace=False)

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

def thin_image(E1, max_iter=100):
    device = E1.device
    
    E1 = E1.clone().float()

    for _ in range(max_iter):
        # Create shifted versions of the image
        P2 = F.pad(E1[:, 1:], (0, 1), "constant", 0)  # Right
        P3 = F.pad(E1[:, :-1], (1, 0), "constant", 0)  # Left
        P4 = F.pad(E1[1:, :], (0, 0, 0, 1), "constant", 0)  # Down
        P5 = F.pad(E1[:-1, :], (0, 0, 1, 0), "constant", 0)  # Up

        # Identify pixels to be removed
        G1 = ((P2 - E1) >= 0) & ((E1 - P3) > 0)
        G2 = ((P4 - E1) >= 0) & ((E1 - P5) > 0)

        C = G1 | G2

        # Update the image
        E1[C] = 0

        # Break if no more changes
        if C.sum() == 0:
            break

    return E1

def match_edge_maps(bmap1, bmap2, maxDist, outlierCost):
    height, width = bmap1.shape

    m1 = np.zeros_like(bmap1)
    m2 = np.zeros_like(bmap2)

    match1 = [[Pixel(-1, -1) for _ in range(width)] for _ in range(height)]
    match2 = [[Pixel(-1, -1) for _ in range(width)] for _ in range(height)]

    r = int(math.ceil(maxDist))

    matchable1 = np.zeros_like(bmap1, dtype=bool)
    matchable2 = np.zeros_like(bmap2, dtype=bool)

    for y1 in range(height):
        for x1 in range(width):
            if not bmap1[y1, x1]:
                continue
            for v in range(-r, r + 1):
                for u in range(-r, r + 1):
                    d2 = u * u + v * v
                    if d2 > maxDist * maxDist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                        continue
                    if not bmap2[y2, x2]:
                        continue
                    matchable1[y1, x1] = True
                    matchable2[y2, x2] = True

    n1, n2 = 0, 0
    nodeToPix1, nodeToPix2 = [], []
    pixToNode1 = np.full((height, width), -1)
    pixToNode2 = np.full((height, width), -1)

    for y in range(height):
        for x in range(width):
            if matchable1[y, x]:
                pixToNode1[y, x] = n1
                nodeToPix1.append(Pixel(x, y))
                n1 += 1
            if matchable2[y, x]:
                pixToNode2[y, x] = n2
                nodeToPix2.append(Pixel(x, y))
                n2 += 1

    edges = []
    for y1 in range(height):
        for x1 in range(width):
            if not matchable1[y1, x1]:
                continue
            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    d2 = u * u + v * v
                    if d2 > maxDist * maxDist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                        continue
                    if not matchable2[y2, x2]:
                        continue
                    edges.append((pixToNode1[y1, x1], pixToNode2[y2, x2], math.sqrt(d2)))

    n = n1 + n2
    nmin, nmax = min(n1, n2), max(n1, n2)

    degree = 6
    d1 = max(0, min(degree, n1 - 1))
    d2 = max(0, min(degree, n2 - 1))
    d3 = min(degree, min(n1, n2))
    dmax = max(d1, max(d2, d3))

    m = len(edges) + d1 * n1 + d2 * n2 + d3 * nmax + n

    if m == 0:
        return 0, m1, m2

    ow = int(math.ceil(outlierCost * 100))

    igraph = np.zeros((m, 3), dtype=int)
    count = 0

    for i, j, w in edges:
        igraph[count] = [i, j, int(round(w * 100))]
        count += 1

    outliers = np.zeros(dmax, dtype=int)

    for i in range(n1):
        outliers[:d1] = k_of_n(d1, n1 - 1)
        for j in outliers:
            if j >= i:
                j += 1
            igraph[count] = [i, n2 + j, ow]
            count += 1

    for j in range(n2):
        outliers[:d2] = k_of_n(d2, n2 - 1)
        for i in outliers:
            if i >= j:
                i += 1
            igraph[count] = [n1 + i, j, ow]
            count += 1

    for i in range(nmax):
        outliers[:d3] = k_of_n(d3, nmin)
        for j in outliers:
            if n1 < n2:
                igraph[count] = [n1 + i, n2 + j, ow]
            else:
                igraph[count] = [n1 + j, n2 + i, ow]
            count += 1

    for i in range(n1):
        igraph[count] = [i, n2 + i, ow]
        count += 1
    for i in range(n2):
        igraph[count] = [n1 + i, i, ow]
        count += 1

    assert count == m

    igraph[:, :2] += 1
    igraph[:, 1] += n

    cost_matrix = np.zeros((2 * n, 2 * n), dtype=int)
    for i, j, c in igraph:
        cost_matrix[i - 1, j - 1] = c

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    overlay_count = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == ow * 100:
            overlay_count += 1

    if overlay_count > 5:
        print(f"WARNING: The match includes {overlay_count} outlier(s) from the perfect match overlay.")

    for i, j in zip(row_ind, col_ind):
        if i < n1 and j >= n and (j - n) < n2:  # 추가된 조건을 확인합니다.
            pix1 = nodeToPix1[i]
            pix2 = nodeToPix2[j - n]
            match1[pix1.y][pix1.x] = pix2
            match2[pix2.y][pix2.x] = pix1


    for y in range(height):
        for x in range(width):
            if bmap1[y, x] and match1[y][x] != Pixel(-1, -1):
                m1[y, x] = match1[y][x].x * height + match1[y][x].y + 1
            if bmap2[y, x] and match2[y][x] != Pixel(-1, -1):
                m2[y, x] = match2[y][x].x * height + match2[y][x].y + 1

    cost = 0.0
    for y in range(height):
        for x in range(width):
            if bmap1[y, x]:
                if match1[y][x] == Pixel(-1, -1):
                    cost += outlierCost
                else:
                    dx = x - match1[y][x].x
                    dy = y - match1[y][x].y
                    cost += 0.5 * math.sqrt(dx * dx + dy * dy)
            if bmap2[y, x]:
                if match2[y][x] == Pixel(-1, -1):
                    cost += outlierCost
                else:
                    dx = x - match2[y][x].x
                    dy = y - match2[y][x].y
                    cost += 0.5 * math.sqrt(dx * dx + dy * dy)

    return cost, m1, m2

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
                    cost, m1, m2 = match_edge_maps(gts[i].cpu().numpy(), E1.cpu().numpy(), maxDist, 100)

                    tp = torch.sum(torch.tensor(m1 > 0).float())
                    fp = torch.sum(torch.tensor(m2 > 0).float()) - tp
                    fn = len(gt_edge_pixels) - tp

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
        precision_np = precision.cpu().numpy()
        recall_np = recall.cpu().numpy()

        # AP를 보간하여 계산
        average_precision = interpolate_ap(precision_np, recall_np)

        print(f"Average Precision for {t}: {average_precision:.4f}")
        print(f"ODS for {t}: {ods:.4f}")

if __name__ == "__main__":
    main()


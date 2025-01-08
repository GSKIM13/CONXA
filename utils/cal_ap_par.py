import os
import torch
import scipy.io
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from torch.nn.functional import interpolate

# 경로 설정
types = ['depth', 'illumination', 'normal', 'reflectance']
pred_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/convnext_att_epoch10000/2000/mat/{}/mat'
gt_dir_template = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/{}/'

# Thresholds
thresholds = torch.linspace(0.01, 0.99, 99, device='cuda')
maxDist = 0.0075  # maxDist 설정
target_size = (321, 481)

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
        
        preds.append(pred_data.squeeze(-1))  # 차원 축소
        gts.append(gt)

    return torch.stack(preds), torch.stack(gts)

def calculate_tp_fp_fn(preds, gts, thresholds, maxDist, image_diagonal):
    tp_sum = torch.zeros(1, device='cuda')
    fp_sum = torch.zeros(1, device='cuda')
    fn_sum = torch.zeros(1, device='cuda')
    tn_sum = torch.zeros(1, device='cuda')

    for threshold in tqdm(thresholds, desc="Thresholds", leave=False):
        binarized_pred = (preds >= threshold).float()
        for i in range(preds.shape[0]):
            gt_edge_pixels = torch.nonzero(gts[i], as_tuple=False).float()
            pred_edge_pixels = torch.nonzero(binarized_pred[i], as_tuple=False).float()

            if len(gt_edge_pixels) == 0:
                fp_sum += torch.sum(binarized_pred[i])
                tn_sum += torch.sum(1 - binarized_pred[i])
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

                tn = torch.sum(1 - binarized_pred[i]) - fn

                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                tn_sum += tn

    return tp_sum, fp_sum, fn_sum, tn_sum

def main():
    pred_dirs = {t: [pred_dir_template.format(t)] for t in types}

    for t in types:
        gt_dir = gt_dir_template.format(t)

        preds, gts = load_and_resize_mat_files(pred_dirs[t], gt_dir)

        image_diagonal = torch.sqrt(torch.tensor(target_size[0]**2 + target_size[1]**2, device='cuda').float()).repeat(preds.shape[0])

        tp_sum, fp_sum, fn_sum, tn_sum = calculate_tp_fp_fn(preds, gts, thresholds, maxDist, image_diagonal)

        # Calculate precision, recall, and average precision
        precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        recall = tp_sum / (tp_sum + fn_sum + 1e-10)
        f_score = 2*precision*recall / (precision+recall+1e-10)
        
        ods = f_score.max().item()
        average_precision = (2 * precision * recall) / (precision + recall + 1e-10)

        print(f"Average Precision for {t}: {average_precision.item():.4f}")
        print(f"ODS for {t}: {ods:.4f}")

if __name__ == "__main__":
    main()

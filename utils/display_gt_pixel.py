import os
import torch
import scipy.io
import matplotlib.pyplot as plt

# 경로 설정
types = ['depth', 'illumination', 'normal', 'reflectance']
pred_dirs = [
    '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/dice40000/dice40000_3000/mat/',
    '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/dice40000/dice40000_5000/mat/'
]
gt_dir = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/'
result_dir = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/pixel_dist'

# 색상 및 투명도 설정
colors = ['blue', 'green']
alpha = 0.5

for type in types:
    plt.figure(figsize=(10, 12))
    
    for i, (pred_dir, color) in enumerate(zip(pred_dirs, colors), start=1):
        result_list = []
        pred_files = sorted([f for f in os.listdir(os.path.join(pred_dir, type, 'mat')) if f.endswith('.mat')])
        gt_files = sorted([f for f in os.listdir(os.path.join(gt_dir, type)) if f.endswith('.mat')])

        for pred_file, gt_file in zip(pred_files, gt_files):
            pred_path = os.path.join(pred_dir, type, 'mat', pred_file)
            pred_tensor = torch.tensor(scipy.io.loadmat(pred_path)['result'])
            gt_path = os.path.join(gt_dir, type, gt_file)
            gt_tensor = torch.tensor(scipy.io.loadmat(gt_path)['groundTruth'][0, 0]['Boundaries'][0, 0])

            result = pred_tensor * gt_tensor
            result = result[result != 0].tolist()
            result_list += result

        plt.subplot(len(pred_dirs), 1, i)
        plt.hist(result_list, bins=100, color=color, alpha=alpha, label=pred_dir.split('/')[-2])
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.ylim(0,50000)
        plt.title(f'Distribution of Non-Zero Elements for {type} - {pred_dir.split("/")[-2]}', fontsize=16)
        plt.legend()

    # 그래프를 파일로 저장
    hist_folder_dir = os.path.join(result_dir, 'dice40000')
    os.makedirs(hist_folder_dir, exist_ok=True)
    plot_dir = os.path.join(hist_folder_dir, f'{type}.png')
    plt.tight_layout()
    plt.savefig(plot_dir)
    plt.close()

import scipy.io
import os 
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from PIL import Image

from scipy.ndimage import binary_dilation

# Paths to the ground truth and result data
ground_truth_path = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/'
result_path_folder = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/cce_att_dice_inv/'

result_paths = os.listdir(result_path_folder)




types = ['depth', 'illumination', 'normal', 'reflectance']
type_ground_truth_path = os.path.join(ground_truth_path, types[0])
 
img_list = os.listdir(type_ground_truth_path)[:-1]



df = pd.DataFrame()

for result_path in result_paths:
    
    folders = os.listdir(os.path.join(result_path_folder, result_path))
    
    iters = [name for name in folders if os.path.isdir(os.path.join(result_path_folder,result_path,name))]
    
    for iter_ in iters:


      for gt_type in types:
          for pred_type in types:
              total_sum = 0
  
              threshold_path = os.path.join(result_path_folder, result_path, iter_, 'mat', pred_type, 'nms_raw-eval', 'eval_bdry.txt')
              with open(threshold_path, 'r') as f:
                  threshold = float(f.readline().strip().split()[0])
                  

  
              # Compute the sum for each image
              for img in img_list:
                  gt_path = os.path.join(ground_truth_path, gt_type, img)
                  mat_data = scipy.io.loadmat(gt_path)
                  gt = mat_data['groundTruth'][0, 0]['Boundaries'][0, 0]
                  gt = torch.from_numpy(gt)
                  
              
                  result_path_ = os.path.join(result_path_folder, result_path, iter_, 'mat', pred_type, 'mat', img)
                  result_data = scipy.io.loadmat(result_path_)
                  result = result_data['result']
                  result = binary_dilation(result, iterations=1)
                  result = torch.from_numpy(result)
              
                  mask = result > threshold
                  total_sum += (gt * mask).sum().item()
  
              df.loc[gt_type, pred_type] = int(total_sum)
  
  
      
      FN = []
      FN.append(iter_)
      FN.append('FN')
      FN.append(458552-df.iloc[0,0])
      FN.append(71717-df.iloc[1,1])
      FN.append(294444-df.iloc[2,2])
      FN.append(254911-df.iloc[3,3])

      FP = []
      FP.append(iter_)
      FP.append('FP')
      FP.append(df.iloc[:,0].sum()-df.iloc[0,0])
      FP.append(df.iloc[:,1].sum()-df.iloc[1,1])
      FP.append(df.iloc[:,2].sum()-df.iloc[2,2])
      FP.append(df.iloc[:,3].sum()-df.iloc[3,3])
  
      DF = [] 
      DF.append(FN) 
      DF.append(FP)
  
      if 'FN_FP_DF' not in locals() or FN_FP_DF is None:
  
          FN_FP_DF = pd.DataFrame(DF)
          FN_FP_DF.columns = ['iter','false', 'depth','illumination','normal','reflectance']
          FN_FP_DF.index = [f'{result_path}_FN',f'{result_path}_FP']
  
      else: 
          new_df = pd.DataFrame(DF)
          new_df.columns = ['iter','false', 'depth','illumination','normal','reflectance']
          new_df.index = [f'{result_path}_FN',f'{result_path}_FP']        
          FN_FP_DF = pd.concat([FN_FP_DF, new_df],axis=0)
        
FN_FP_DF.to_csv('./utils/fn_fp_count_loss_1112_raw.csv')

pd.set_option('display.float_format', '{:.4f}'.format)  
print(tabulate(FN_FP_DF, headers='keys', tablefmt='psql', showindex=True))  

data = FN_FP_DF

data.reset_index(inplace=True)

data.columns = ['data','iter','false', 'depth','illumination','normal','reflectance']


          
data['category'] = data['data'].str.extract(r'^(dice|inv_dice|cce|att)')

data = data.dropna(subset=['category'])


grouped_averages = data.groupby(['category', 'false'])[['depth', 'illumination', 'normal', 'reflectance']].mean()

grouped_averages = grouped_averages.astype(int)

grouped_averages['sum'] = grouped_averages[['depth', 'illumination', 'normal', 'reflectance']].sum(axis=1)


grouped_averages.reset_index(inplace=True)


        
grouped_averages.to_csv(f'./utils/1112/fn_fp_summary_raw.csv')

pd.set_option('display.float_format', '{:.4f}'.format)  
print(tabulate(grouped_averages, headers='keys', tablefmt='psql', showindex=True))  
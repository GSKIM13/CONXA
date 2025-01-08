
import os 

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np 
 
import matplotlib
matplotlib.use('Agg')  
from PIL import Image

from matplotlib.ticker import FuncFormatter 



# Paths to the ground truth and result data
ground_truth_path = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/data/BSDS-RIND_ORI/testgt/'
result_path_folder = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/cce_att_dice_inv/'

result_paths = os.listdir(result_path_folder)




types = ['depth', 'illumination', 'normal', 'reflectance']
type_ground_truth_path = os.path.join(ground_truth_path, types[0])
 
img_list = os.listdir(type_ground_truth_path)[:-1]



df = pd.DataFrame()

final_result = []

for result_path in result_paths:
    
    folders = os.listdir(os.path.join(result_path_folder, result_path))
    
    iters = [name for name in folders if os.path.isdir(os.path.join(result_path_folder,result_path,name))]
    
    
    
    for iter_ in iters:
      data1 = []
      data1.append(result_path)
      data1.append(iter_)
      for pred_type in types:
      
        
      
        total_FN = 0
        total_FP = 0
        total_TP = 0

        threshold_path = os.path.join(result_path_folder, result_path, iter_, 'mat', pred_type, 'nms_raw-eval', 'eval_bdry.txt')
        with open(threshold_path, 'r') as f:
            threshold = float(f.readline().strip().split()[0])
            

            
        # Compute the sum for each image
        for img in img_list:
            result_path_ = os.path.join(result_path_folder, result_path, iter_, 'mat', pred_type, 'nms_raw-eval', img[:-4]+'_ev1.txt')
            
            with open(result_path_, 'r') as fr:
              lines = fr.readlines()
            
            data = [line.split() for line in lines]
            result = data[int(threshold*100)-1] 

            
            total_FN += (int(result[2]) - int(result[1]))
            total_FP += (int(result[4]) - int(result[3]))
            total_TP += int(result[2])
        

        data1.append(total_FN)
        data1.append(total_FP)
        
    final_result.append(data1)
         
df = pd.DataFrame(final_result)  

df.columns=['result_path','iter','depth_FN','depth_FP','illu_FN','illu_FP','nor_FN','nor_FP','ref_FN','ref_FP']

df['category'] = df['result_path'].str.extract(r'^(dice|inv_dice|cce|att(?:_dice|_inv_dice)?)') 

grouped_averages = df.groupby(['category']).mean().astype(int) 

grouped_averages['FN'] = grouped_averages['depth_FN']+grouped_averages['illu_FN']+grouped_averages['nor_FN']+grouped_averages['ref_FN'] 
 
grouped_averages['FP'] = grouped_averages['depth_FP']+grouped_averages['illu_FP']+grouped_averages['nor_FP']+grouped_averages['ref_FP'] 

desired_order = ['cce','att','dice','inv_dice','att_dice','att_inv_dice']

grouped_averages.index = pd.Categorical(grouped_averages.index, categories=desired_order, ordered=True)

grouped_averages = grouped_averages.sort_index()

grouped_averages.index = ['CCE','Att','Dice','I-Dice','Att_Dice','Att_I-Dice']

fig, ax = plt.subplots(figsize=(20, 15)) 

df = grouped_averages

plt.rcParams['font.family'] = 'Palatino Linotype'


fn_col = [0,2,4,6,8] 
fp_col = [1,3,5,7,9]
 
df_fn = df.iloc[:,fn_col]
df_fp = df.iloc[:,fp_col]

df_fn.reset_index(inplace=True)
df_fp.reset_index(inplace=True)

df_fn.columns = ['Loss','depth','illumination','normal','reflectance','sum']
df_fp.columns = ['Loss','depth','illumination','normal','reflectance','sum']


bar_width = 0.4
index = range(len(df_fn))



depth_color_fn = '#1f77b4'       
depth_color_fp = '#aec7e8'       
illumination_color_fn = '#ff7f0e' 
illumination_color_fp = '#ffbb78' 
normal_color_fn = '#2ca02c'      
normal_color_fp = '#98df8a'      
reflectance_color_fn = '#d62728' 
reflectance_color_fp = '#ff9896'  


p1 = ax.bar(index, df_fn['depth'], bar_width, label='FN - Depth', color=depth_color_fn, edgecolor='black')
p2 = ax.bar(index, df_fn['illumination'], bar_width, bottom=df_fn['depth'], label='FN - Illumination', color=illumination_color_fn, edgecolor='black')
p3 = ax.bar(index, df_fn['normal'], bar_width, bottom=df_fn['depth'] + df_fn['illumination'], label='FN - Normal', color=normal_color_fn, edgecolor='black')
p4 = ax.bar(index, df_fn['reflectance'], bar_width, bottom=df_fn['depth'] + df_fn['illumination'] + df_fn['normal'], label='FN - Reflectance', color=reflectance_color_fn, edgecolor='black')


index_fp = [i + bar_width for i in index]
p5 = ax.bar(index_fp, df_fp['depth'], bar_width, label='FP - Depth', color=depth_color_fp, edgecolor='black')
p6 = ax.bar(index_fp, df_fp['illumination'], bar_width, bottom=df_fp['depth'], label='FP - Illumination', color=illumination_color_fp, edgecolor='black')
p7 = ax.bar(index_fp, df_fp['normal'], bar_width, bottom=df_fp['depth'] + df_fp['illumination'], label='FP - Normal', color=normal_color_fp, edgecolor='black')
p8 = ax.bar(index_fp, df_fp['reflectance'], bar_width, bottom=df_fp['depth'] + df_fp['illumination'] + df_fp['normal'], label='FP - Reflectance', color=reflectance_color_fp, edgecolor='black')


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  
            bar.get_y() + height / 2, 
            f'{int(height):,}',  
            ha='center',  
            va='center', 
            fontsize=19,  
            color='black'
        )

add_labels(p1)
add_labels(p2)
add_labels(p3)
add_labels(p4)
add_labels(p5)
add_labels(p6)
add_labels(p7)
add_labels(p8)


for i, rect in enumerate(index):

    ax.text(
        rect, 
        df_fn.iloc[i]['depth'] + df_fn.iloc[i]['illumination'] + df_fn.iloc[i]['normal'] + df_fn.iloc[i]['reflectance'] + 5000,  
        f"{int(df_fn.iloc[i]['sum']):,}",
        ha='center',
        va='bottom',
        fontsize=19,
        color='black'
    )

    ax.text(
        rect + bar_width, 
        df_fp.iloc[i]['depth'] + df_fp.iloc[i]['illumination'] + df_fp.iloc[i]['normal'] + df_fp.iloc[i]['reflectance'] + 5000,  
        f"{int(df_fp.iloc[i]['sum']):,}",
        ha='center', 
        va='bottom',
        fontsize=19, 
        color='black'
    )

 
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))


ax.set_xlabel('Loss Function', fontsize=27)
ax.set_ylabel('False Counts', fontsize=27)
ax.set_xticks([r + bar_width / 2 for r in range(len(df_fn))])
ax.set_xticklabels(df_fn['Loss'], fontsize=27)
ax.tick_params(axis='y', labelsize=27)   

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fig_raw.png')



pd.set_option('display.float_format', '{:.4f}'.format)  
print(tabulate(grouped_averages, headers='keys', tablefmt='psql', showindex=True))  

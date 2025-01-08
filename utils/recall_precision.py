import os
import pandas as pd
from tabulate import tabulate

# DataFrame 초기화
df = pd.DataFrame(columns=[
    'data', 'ref_r', 'ref_p', 'ref_f',
    'illu_r', 'illu_p', 'illu_f',
    'nor_r', 'nor_p', 'nor_f',
    'dep_r', 'dep_p', 'dep_f'
])

data_dir = '/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/work_dirs/iter4' 


# 데이터 폴더 추출
data_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


types = ['reflectance', 'illumination', 'normal', 'depth']

# 데이터 수집 및 DataFrame에 추가
for data in data_folders:
    data_path = os.path.join(data_dir, data)
    iterations = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for iteration in iterations:
        iter_path = os.path.join(data_path, iteration)
        
        # 경로에서 마지막 두 디렉토리만 추출
        relative_path = os.path.relpath(iter_path, data_dir)
        
        row = {'data': relative_path}  # 경로에서 마지막 두 디렉토리만 저장
        
        for type_ in types:
            category_path = os.path.join(iter_path, 'mat', type_, 'nms-eval', 'eval_bdry.txt')
            
            try:
                with open(category_path, 'r') as f:
                    data_line = f.readline()
                    recall = round(float(data_line.split()[1]),3)
                    precision = round(float(data_line.split()[2]),3)
                    fscore = round(float(data_line.split()[3]),3)
                    
                    # 열 이름 매핑
                    if type_ == 'reflectance':
                        row['ref_r'] = recall
                        row['ref_p'] = precision
                        row['ref_f'] = fscore
                    elif type_ == 'illumination':
                        row['illu_r'] = recall
                        row['illu_p'] = precision
                        row['illu_f'] = fscore
                    elif type_ == 'normal':
                        row['nor_r'] = recall
                        row['nor_p'] = precision
                        row['nor_f'] = fscore
                    elif type_ == 'depth': 
                        row['dep_r'] = recall
                        row['dep_p'] = precision 
                        row['dep_f'] = fscore
            
            except FileNotFoundError:
                print(f"File not found: {category_path}. Skipping...")
                continue
        
        # DataFrame에 행 추가
        df = df.append(row, ignore_index=True)
    

# 보기 좋은 형식으로 출력
df.sort_values(by = 'data', inplace=True)
df.to_csv('recall_precision_loss_iter.csv')
pd.set_option('display.float_format', '{:.4f}'.format)  # 소수점 4자리로 맞춤
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))  # DataFrame을 표 형식으로 출력



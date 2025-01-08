

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd 

# 파일이 저장된 디렉토리 경로
directory_path = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/test/1/"

# 모든 .txt 파일 경로를 가져옴
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

# 파일 개수 확인
num_files = len(file_paths)
print(f"총 {num_files}개의 파일을 찾았습니다.")

# 첫 번째 파일을 읽어 크기 확인
sample_array = np.genfromtxt(file_paths[0], delimiter=',')
rows, cols = sample_array.shape

# 3차원 배열 초기화
data_3d = np.zeros((num_files, rows, cols))

# 모든 파일을 읽어 3차원 배열에 저장
for i, file_path in enumerate(file_paths):
    data_3d[i, :, :] = np.genfromtxt(file_path, delimiter=',')

# 평균을 내어 2차원 배열로 변환
average_data = np.mean(data_3d, axis=0)

df = pd.DataFrame(average_data)
#df.to_csv('./display_best/df.csv')

# 256x128 데이터를 4x4 데이터로 변환하기 위해 블록 합계 계산
block_size_row = rows // 4
block_size_col = cols // 4

resampled_data = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        block = average_data[i*block_size_row:(i+1)*block_size_row, j*block_size_col:(j+1)*block_size_col]
        resampled_data[i, j] = block.sum()

# 히트맵으로 시각화 (흑백 컬러맵 사용)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(resampled_data/64, cmap='Greys_r', annot=False)
#ax = sns.heatmap(resampled_data/64, cmap='Greys_r', annot=True, fmt=".2f")
plt.title('Heatmap of Summed Data (4x4)')
plt.savefig('/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/test/fig_summed_1_4_4.png')
plt.show()


'''

# 합산된 데이터를 numpy 파일로 저장
#output_file_path = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display_best/average_data_resampled_summed.npy"
#np.save(output_file_path, resampled_data)
#print(f"리샘플링되고 합산된 평균 데이터가 저장되었습니다: {output_file_path}")

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 파일이 저장된 디렉토리 경로
directory_path = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display_best/"
directory_path_ = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/"

# 모든 .txt 파일 경로를 가져옴
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

# 파일 개수 확인
num_files = len(file_paths)
print(f"총 {num_files}개의 파일을 찾았습니다.")

# 첫 번째 파일을 읽어 크기 확인
sample_array = np.genfromtxt(file_paths[0], delimiter=',')
rows, cols = sample_array.shape

# 모든 파일에 대해 작업 수행
for file_index, file_path in enumerate(file_paths):
    data = np.genfromtxt(file_path, delimiter=',')

    # 256x128 데이터를 4x4 데이터로 변환하기 위해 블록 합계 계산
    block_size_row = rows // 4
    block_size_col = cols // 4

    resampled_data = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            block = data[i*block_size_row:(i+1)*block_size_row, j*block_size_col:(j+1)*block_size_col]
            resampled_data[i, j] = block.sum()

    # 히트맵으로 시각화 (흑백 컬러맵 사용)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(resampled_data / 64, cmap='Greys_r', annot=True, fmt=".2f")
    plt.title(f'Heatmap of Summed Data (4x4) - File {file_index+1}')
    
    # 파일 이름을 기반으로 저장
    output_path = os.path.join(directory_path_, f'fig_resampled_summed_{file_index+1}.png')
    plt.savefig(output_path)
    plt.close()

print("모든 히트맵이 생성되었습니다.")


'''
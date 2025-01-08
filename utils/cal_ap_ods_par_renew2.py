import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# 파일이 저장된 디렉토리 경로
directory_path = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/"

# 모든 .txt 파일 경로를 가져옴
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

# 파일 개수 확인
num_files = len(file_paths)
print(f"총 {num_files}개의 파일을 찾았습니다.")

# 첫 번째 파일을 읽어 크기 확인
sample_array = np.genfromtxt(file_paths[0], delimiter=',')
rows, cols = sample_array.shape

# 3차원 배열 초기화 (800 x 128 x 256)
data_3d = np.zeros((num_files, rows, cols))

# 모든 파일을 읽어 3차원 배열에 저장
for i, file_path in enumerate(file_paths):
    data_3d[i, :, :] = np.genfromtxt(file_path, delimiter=',')

# 평균을 내어 2차원 배열로 변환
average_data = np.mean(data_3d, axis=0)

# 히트맵으로 시각화
plt.imshow(average_data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap of Averaged Data')
plt.savefig('/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/fig.png')
plt.show()

# 평균 데이터를 numpy 파일로 저장
output_file_path = "/home/gwangsoo13kim/EdgeSementic/EDTER_TriDecTr_GS/display/average_data.npy"
np.save(output_file_path, average_data)
print(f"평균 데이터가 저장되었습니다: {output_file_path}")

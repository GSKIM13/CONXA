import torch

# 모델 파일 경로
file_path = '/home/gwangsoo13kim/EdgeSementic/SWIN-RIND/run/swin-rind/edge_experiment_21/epoch_10_checkpoint.pth.tar'

# 모델 파라미터 로드
model_params = torch.load(file_path)

model_params = model_params['state_dict']

# 각 파라미터 텐서의 평균 값 계산
param_means = {name: param.mean().item() for name, param in model_params.items()}

# 결과 출력
for name, mean in param_means.items():
    print(f"Parameter: {name}, Mean: {mean}")

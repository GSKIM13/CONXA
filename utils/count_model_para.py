import torch
import os
import sys

# .pth 파일 경로
pth_file_path = '/home/gwangsoo13kim/EdgeSementic/ECT/pretrained_models/full_version.pth.tar'

# .pth 파일 용량 확인
file_size = os.path.getsize(pth_file_path)
print(f'.pth 파일 용량: {file_size / (1024 * 1024):.2f} MB')

# .pth 파일에서 전체 데이터를 CPU로 로드
loaded_data = torch.load(pth_file_path, map_location=torch.device('cpu'))

# 각 키와 데이터 타입 및 크기 출력
total_additional_size = 0

def get_size(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    elif isinstance(value, (list, tuple)):
        return sum(get_size(v) for v in value)
    elif isinstance(value, dict):
        return sum(get_size(v) for v in value.values())
    else:
        return sys.getsizeof(value)

for key, value in loaded_data.items():
    size_in_bytes = get_size(value)
    total_additional_size += size_in_bytes

# 전체 파라미터 수 계산 및 각각의 파라미터에 대한 정보 출력
model_state_dict = loaded_data['state_dict']
total_params = 0
backbone_params = 0
non_backbone_params = 0

for key, param in model_state_dict.items():
    if isinstance(param, torch.Tensor):
        param_numel = param.numel()
        param_size = param_numel * param.element_size()
        print(f'name : {key}, count : {param_numel}')
        total_params += param_numel
        
        if 'backbone' in key:
            backbone_params += param_numel
        else:
            non_backbone_params += param_numel
        

# 출력
print(f'총 파라미터 수: {total_params}')
print(f'backbone 파라미터 수: {backbone_params}')
print(f'나머지 파라미터 수: {non_backbone_params}')

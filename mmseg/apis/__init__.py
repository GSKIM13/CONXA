from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test, multi_gpu_test_sbd, multi_gpu_test_city
from .train import get_root_logger, set_random_seed, train_segmentor
from .train_city import get_root_logger, set_random_seed, train_segmentor_city
from .train_sbd import get_root_logger, set_random_seed, train_segmentor_sbd
from .train_local8x8 import get_root_logger, set_random_seed, train_segmentor_local8x8

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', multi_gpu_test_sbd, multi_gpu_test_city,'single_gpu_test',
    'show_result_pyplot','train_segmentor_local8x8'
]

import mmcv
import numpy as np
from numpy import random
from torchvision import transforms
import torch

from ..builder import PIPELINES

import cv2

from PIL import Image, ImageOps

@PIPELINES.register_module()
class PadToMinimumSizeWithSeg(object):
    def __init__(self, min_size=(320, 320), pad_val=0, seg_pad_val=0):
        self.min_size = min_size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results):
        img = results['img']
        h, w = img.shape[:2]

        if h >= self.min_size[0] and w >= self.min_size[1]:
            results['img'] = img
            results['pad_shape'] = img.shape
        else:
            pad_h = max(self.min_size[0] - h, 0)
            pad_w = max(self.min_size[1] - w, 0)
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_val)
            results['img'] = padded_img
            results['pad_shape'] = padded_img.shape
    
    def _pad_seg(self, results):
        """Pad masks to ensure the size matches the padded image size."""
        for key in results.get('seg_fields', []):
            mask = results[key]
            h, w, c = mask.shape  # Get the shape of the mask
    
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8)  # Convert boolean mask to uint8
    
            if h >= self.min_size[0] and w >= self.min_size[1]:
                results[key] = mask
            else:
                pad_h = max(self.min_size[0] - h, 0)
                pad_w = max(self.min_size[1] - w, 0)
                top, bottom = pad_h // 2, pad_h - pad_h // 2
                left, right = pad_w // 2, pad_w - pad_w // 2
                
                #print(mask.max())
    
                # Pad each channel separately
                padded_channels = []
                for i in range(c):  # Iterate over each channel
                    channel = mask[:, :, i]
                    padded_channel = cv2.copyMakeBorder(
                        channel, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.seg_pad_val
                    )
                    padded_channels.append(padded_channel)
    
                # Stack channels back together
                padded_mask = np.stack(padded_channels, axis=-1)
                results[key] = padded_mask
                
                



    def __call__(self, results):
        self._pad_img(results)
        if 'seg_fields' in results:
            self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_size={self.min_size}, pad_val={self.pad_val}, seg_pad_val={self.seg_pad_val})'
        return repr_str



@PIPELINES.register_module()
class PadToMinimumSize(object):
    """Pad the image to ensure the size is at least min_size.

    Args:
        min_size (tuple): Minimum size of the padded image.
        pad_val (float, optional): Padding value for images. Default: 0.
    """

    def __init__(self, min_size=(320, 320), pad_val=0):
        self.min_size = min_size
        self.pad_val = pad_val

    def _pad_img(self, results):
        """Pad images to ensure the size is at least min_size."""
        img = results['img']
        h, w = img.shape[:2]
        
        print(f'h,w : {h,w}')

        if h >= self.min_size[0] and w >= self.min_size[1]:
            results['img'] = img
            results['pad_shape'] = img.shape
            print('pass')
        else:
            pad_h = max(self.min_size[0] - h, 0)
            pad_w = max(self.min_size[1] - w, 0)
            
            print(f'pad_size : {padded_img.shape}')
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.pad_val)
            results['img'] = padded_img
            results['pad_shape'] = padded_img.shape
            
        
        

    def __call__(self, results):
        """Call function to pad images."""
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_size={self.min_size}, pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 dataset='bsds',
                 multi_scale_test=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.dataset = dataset
        self.multi_scale_test = multi_scale_test

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        #print(self.dataset)
        if self.keep_ratio:
            if self.multi_scale_test or self.dataset == 'pascal_context' or self.dataset == 'cocostuff':
                #print("multi_scale")
                min_short = {'cityscapes':768, 'ade20k': 512, 'pascal_context': 480, 'bsds': 321,'multicue':720}
                if min(results['scale']) < min_short[self.dataset]:
                    new_short = min_short[self.dataset]
                else:
                    new_short = results['scale'][0]

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)
                
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class PadBSDS(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        #img = results['img']
        #r, c, k = img.shape
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val, padding_mode='symmetric')
        #elif self.size_divisor is not None:
        #    padded_img = mmcv.impad_to_multiple(results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val,
                padding_mode='symmetric'
            )

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        #self.pad_val = np.mean(results['img'], axis=(0, 1))  ##(3,)
        #self.seg_pad_val = np.array([0]).reshape(1, )
        self._pad_img(results)
        self._pad_seg(results)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class NormalizeBSDS(object):
    """NormalizeBSDS the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = self.trans(results['img'])
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        results['gt_semantic_seg'] = torch.from_numpy(results['gt_semantic_seg']).float().unsqueeze(0)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class NormalizeBSDSTest(object):
    """NormalizeBSDSTest the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = self.trans(results['img'])
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        #img_path = results['img_info']['filename']
        #img_name = img_path.split('/')[-1]
        #results['img_id'] = img_name.split('.')[0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class NormalizeHHA(object):
    """NormalizeHHA the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        #normalize = transforms.Normalize(mean=self.mean, std=self.std)
        #self.trans = transforms.Compose([transforms.ToTensor(), normalize])

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        #results['img'] = self.trans(results['img'])
        img = np.array(results['img'],dtype=np.float32)
        img -= self.mean
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img.copy()).float()
        results['img'] = img
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        results['gt_semantic_seg'] = torch.from_numpy(results['gt_semantic_seg']).float().unsqueeze(0)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class NormalizeHHATest(object):
    """NormalizeHHA the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        #normalize = transforms.Normalize(mean=self.mean, std=self.std)
        #self.trans = transforms.Compose([transforms.ToTensor(), normalize])

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        #results['img'] = self.trans(results['img'])
        img = np.array(results['img'],dtype=np.float32)
        img -= self.mean
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img.copy()).float()
        results['img'] = img
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class RandomCropTrain(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        #print("here???")
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        #print("===")
        #print(np.asarray(results['gt_semantic_seg']).shape)
        #print(results['img'].shape)
        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        
        #print("===")
        #print(np.asarray(results['gt_semantic_seg']).shape)
        #print(results['img'].shape)
        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        #print(results['img'].shape)
        #print(np.asarray(results['gt_semantic_seg']).shape)
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomScaleCrop(object):
    """Randomly scales, crops, and optionally flips the image and segmentation map.

    Args:
        base_size (int): Base size for scaling.
        crop_size (int): Desired crop size.
        scale_range (tuple): Tuple of two floats specifying the scaling range.
            Default is (0.75, 2.0).
        flip_prob (float): Probability of performing horizontal flip. Default is 0.5.
        ignore_index (int): Index to ignore in segmentation map during loss computation.
            Default is 255.
    """

    def __init__(self,

                 crop_size,
                 scale_range=(0.75, 2.0),
                 flip_prob=0.5,
                 ignore_index=255, base_size = 520):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.ignore_index = ignore_index

        # Mean values for image normalization (ImageNet means)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def random_flip(self, img, mask):
        """Randomly flip the image and mask horizontally."""
        


        if random.random() < self.flip_prob:
            img = np.fliplr(img)
            mask = np.fliplr(mask)


        return img, mask

    def random_scale(self, img, mask):
        """Randomly scale the image and mask."""
        

        short_size = random.randint(int(self.base_size * self.scale_range[0]),
                                    int(self.base_size * self.scale_range[1]))
        h, w = img.shape[:2]
        if h > w:
            ow = short_size
            oh = int(h * short_size / w)
        else:
            oh = short_size
            ow = int(w * short_size / h)
        
        # Convert mask from bool to uint8
        if mask.dtype == np.bool_ or mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255  # Convert boolean to uint8 (0 and 255)
                 
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((ow, oh), Image.BILINEAR)
        img = np.array(img_pil)
        
        resized_mask_list = []
        for i in range(mask.shape[2]):  # Iterate over the 19 channels (classes)
            single_class_mask = mask[:, :, i]  # Extract the 2D mask for the i-th class
            mask_pil = Image.fromarray(single_class_mask)  # Convert to PIL Image
            mask_pil = mask_pil.resize((ow, oh), Image.NEAREST)  # Resize using nearest neighbor
            resized_single_class_mask = np.array(mask_pil)  # Convert back to numpy array
            resized_mask_list.append(resized_single_class_mask)  # Append to list
    
        # Stack resized masks along the third dimension to form the final mask
        mask = np.stack(resized_mask_list, axis=-1)
        
        return img, mask

    def pad(self, img, mask):
        """Pad image and mask to the desired crop size."""
        h, w = img.shape[:2]  
        pad_h = max(self.crop_size[0] - h, 0)
        pad_w = max(self.crop_size[1] - w, 0)
    
        if pad_w > 0 or pad_h > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=[(int(255 * m),) for m in self.mean])
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.ignore_index)
    
        return img, mask

    
    def random_crop(self, img, mask):
        """Randomly crop the image and mask."""
        h, w = img.shape[:2]
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img[y1:y1 + self.crop_size[0], x1:x1 + self.crop_size[1], :]
        mask = mask[y1:y1 + self.crop_size[0], x1:x1 + self.crop_size[1]]
        return img, mask

    def normalize(self, img):
        """Normalize the image."""
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img

    def to_tensor(self, img, mask):
        """Convert image and mask to torch tensors."""
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = np.array(mask, dtype=np.uint8)
        return img, torch.from_numpy(mask).long()

    def __call__(self, results):
        """Apply transformations to the image and mask."""

        img = results['img']
        mask = results['gt_semantic_seg']

        # Apply random flip
        img, mask = self.random_flip(img, mask)

        # Apply random scale
        img, mask = self.random_scale(img, mask)

        # Apply padding if necessary
        img, mask = self.pad(img, mask)

        # Apply random crop
        img, mask = self.random_crop(img, mask)

        # Normalize image
        img = self.normalize(img)
        
        
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        #mask = np.transpose(mask, (2,0,1))
        mask = (mask / 255).astype(np.float32)


        # Update results
        results['img'] = img
        results['gt_semantic_seg'] = mask
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(base_size={self.base_size}, crop_size={self.crop_size}, '
        repr_str += f'scale_range={self.scale_range}, flip_prob={self.flip_prob}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str
    

@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class BSDSResize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 dataset='bsds',
                 multi_scale_test=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.dataset = dataset
        self.multi_scale_test = multi_scale_test


    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # print(self.dataset)
        if self.keep_ratio:
            if self.multi_scale_test or self.dataset == 'pascal_context' or self.dataset == 'cocostuff':
                # print("multi_scale")
                min_short = {'cityscapes': 768, 'ade20k': 512, 'pascal_context': 480, 'bsds': 160}
                if min(results['scale']) < min_short[self.dataset]:
                    new_short = min_short[self.dataset]
                else:
                    new_short = results['scale'][0]

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

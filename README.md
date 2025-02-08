# CONXA
> CONXA: A CONvnext and CROSS-Attention Combination Network for Semantic Edge Detection

## Environment Setup
Our Project is developed based on MMsegmentation

The full script for settng up CONXA with conda is modifying [EDTER](https://github.com/MengyangPu/EDTER), [SETR](https://github.com/fudan-zvg/SETR#linux).


```
conda create -n conxa python=3.7 -y
conda activate conxa
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
cd CONXA
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
conda install -c conda-forge rust
pip install yapf==0.32.0
pip install h5py scipy pandas timm
```

## Dataset Preparation 

**BSDS-RIND**

Download Benchmark Dataset (BSDS-RIND.zip, testgt.zip) from [RINDNet](https://github.com/MengyangPu/RINDNet)

Move the files as shown below.


**BSDS-RIND.zip**<br/>
BSDS-RIND/Augmentation/Aug_JPEGIMages -> data/BSDS-RIND_ORI/aug_data<br/>
BSDS-RIND/Augmentation/Aug_HDF5EdgeOriLabel -> data/BSDS-RIND_ORI/aug_gt<br/>
BSDS-RIND/test -> data/BSDS-RIND_ORI/test<br/>

**testgt.zip**<br/>
testgt -> data/BSDS-RIND_ORI/testgt<br/>
```
|-- data
    |-- BSDS-RIND_ORI
        |-- ImageSets
        |   |-- train_pair.txt
        |-- train
        |   |-- aug_data
        |   |-- aug_gt
        |-- test
        |   |-- 2018.jpg
        |-- testgt
        |   |-- depth
        |   |-- illumination
        |   |-- normal
        |   |-- reflectnace
        ......
```
**SBD,CITYSCAPES**<br/>
Follow Sequence provided by [SEAL](https://github.com/Chrisding/seal)

```
|-- data
    |-- sbd-preprocess
        |-- data_proc
        |-- gt_eval
    |-- cityscapes-preprocess
        |-- data_proc
        |-- gt_eval

```
    
## Training 
We assume that you'are in CONXA Folder.<br/>
**For BSDS-RIND Dataset**<br/>
```
bash ./tools/dist_train.sh configs/bsds-rind/CONXA_BSDS-RIND.py 1 --work-dir work_dirs/{result_name}
```
**For SBD Dataset**<br/>
**currently not available, Planned Update**
```
bash ./tools/dist_train.sh configs/sbd/CONXA_SBD_CLS.py 1 --work-dir work_dirs/{result_name}
```

**For CITYSCAEPS Dataset**<br/>
**currently not available, Planned Update**
```
bash ./tools/dist_train.sh configs/cityscapes/CONXA_CITY.py 1 --work-dir work_dirs/{result_name}
```

## Test 
We assume that you'are in CONXA Folder.<br/>
**For BSDS-RIND Dataset**<br/>
```
test.py --checkpoint pretrained/iter_3000.pth --configs configs/bsds-rind/ConvNeXt_V2_case8.py --tmpdir result
```

**For SBD Dataset**<br/>
**currently not available, Planned Update**
```
test.py --checkpoint pretrained/iter_20000.pth --configs configs/sbd/CONXA_SBD_CLS.py --tmpdir result
```

**For CITYSCAEPS Dataset**<br/>
**currently not available, Planned Update**
```
test.py --checkpoint pretrained/iter_80000.pth --configs configs/cityscapes/CONXA_CITY.py --tmpdir result
```

## Pretrained Weight
You can download Pretrained Weight from [here](https://drive.google.com/drive/folders/1OR7zOD2zXK1Kbb35n4bx2OMX2TMIcx6S?usp=drive_link)

## Evaluation 
**BSDS-RIND**<br/>
You can download matlab file from [RINDNet](https://github.com/MengyangPu/RINDNet/tree/main)
```
cd eval
run eval.m
```

**SBD, CITYSCAEPS Dataset**<br/>
Follow Sequence provided by [SEAL](https://github.com/Chrisding/seal)

## Acknowledgments
- Thanks to previous open-sourced github repo:<br/>
  [RINDNet](https://github.com/MengyangPu/RINDNet/tree/main)<br/>
  [EDTER](https://github.com/MengyangPu/EDTER)<br/>
  [DFF](https://github.com/Lavender105/DFF)<br/>
  [SEAL](https://github.com/Chrisding/seal)<br/>
  [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2/tree/main)<br/>
  [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)<br/>

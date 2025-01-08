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
```

## Dataset Preparation (BSDS-RIND)

Download Benchmark Dataset (BSDS-RIND.zip, testgt.zip) from [RINDNet](https://github.com/MengyangPu/RINDNet)

Move the files below sequence

BSDS-RIND/Augmentation/Aug_JPEGIMages -> data/BSDS-RIND_ORI/aug_data
BSDS-RIND/Augmentation/Aug_HDF5EdgeOriLabel -> data/BSDS-RIND_ORI/aug_gt
BSDS-RIND/test -> data/BSDS-RIND_ORI/test

testgt -> data/BSDS-RIND_ORI/testgt
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

## Training 
We assume that you'are in CONXA Folder.
```
bash ./tools/dist_train.sh configs/bsds-rind/ConvNeXt_V2_case8_bn.py
```

## Test 
We assume that you'are in CONXA Folder.
```
test.py --checkpoint pretrained/iter_3000.pth --configs configs/bsds-rind/ConvNeXt_V2_case8_bn.py --tmpdir result
```

## Pretrained Weight
You can download Pretrained Weight from [here]()

## Evaluation 
You can download matlab file from [RINDNet](https://github.com/MengyangPu/RINDNet/tree/main)
```
cd eval
run eval.m
```

## Acknowledgments
- Thanks to previous open-sourced github repo:<br/>
  [RINDNet](https://github.com/MengyangPu/RINDNet/tree/main)
  [EDTER](https://github.com/MengyangPu/EDTER)
  [DFF](https://github.com/Lavender105/DFF)
  [SEAL](https://github.com/Chrisding/seal)

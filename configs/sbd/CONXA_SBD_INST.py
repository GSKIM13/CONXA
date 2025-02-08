norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CONXA_Backbone_SBD',
        mla_channels=256,
        model_name='convnext_v2_large_224',
        mla_index=(0, 1, 2, 3),
        norm_cfg=dict(type='LN', requires_grad=True),
        drop_rate=0.0,
        category_emb_dim=1280,
        embed_dim = 192,
        scale=1280),
    decode_head=dict(
        type='CONXA_Head_SBD',
        in_channels=1024,
        channels=512,
        img_size=320,
        middle_channels=32,
        head_channels=16,
        num_classes=20,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CONXALoss', use_sigmoid=True, attention_coeff=1.0, beta = 8, gamma =0.5, dice_coeff = 0, inv_dice_coeff = 10000000),
        category_emb_dim=1280),
   
        
          
)
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
dataset_type = 'PascalVOCDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_SBD'),
    dict(type='RandomCropTrain', crop_size=(320, 320), cat_max_ratio=0.75),
    dict(type='PadBSDS', size=(320, 320), pad_val=0, seg_pad_val=255),
    dict(
        type='NormalizeBSDS',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 320),
        flip=False,
        transforms=[
            dict(
                type='NormalizeBSDSTest',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=10,
    train=dict(
        type='PascalVOCDataset',
        data_root='/home/gwangsoo13kim/EdgeSementic/CONXA/data/sbd-preprocess/data_proc/',
        img_dir='',
        ann_dir='',
        split='trainvalaug_inst_orig_.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations_SBD'),
            dict(type='PadToMinimumSizeWithSeg'),
            dict(
                type='RandomCropTrain',
                crop_size=(320, 320),
                cat_max_ratio=0.75),
            # dict(type='PadBSDS', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(
                type='NormalizeBSDS',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='PascalVOCDataset',
        data_root='/home/gwangsoo13kim/EdgeSementic/CONXA/data/sbd-preprocess/data_proc/',
        img_dir='',
        ann_dir='',
        split='test_inst_orig_.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 320),
                flip=False,
                transforms=[
                    dict(
                        type='NormalizeBSDSTest',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalVOCDataset',
        data_root='/home/gwangsoo13kim/EdgeSementic/CONXA/data/sbd-preprocess/data_proc/',
        img_dir='',
        ann_dir='',
        split='test_inst_orig_.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 320),
                flip=False,
                transforms=[
                    dict(
                        type='NormalizeBSDSTest',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=5e-5,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
    
optimizer_config = dict()
total_iters = 25000
checkpoint_config = dict(by_epoch=False, interval=25000)
evaluation = dict(interval=25000, metric='mIoU')

lr_config = dict(policy='fixed')
find_unused_parameters = True
work_dir = 'work_dirs_case14_256'
gpu_ids = range(0, 1)



_base_ = '../../_base_/default_runtime.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN', graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=11, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'dataset/mmad_plus_stgcn_dataset/mmad3d_mmaction2.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='mmad', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=179),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='mmad', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=179, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='mmad', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=179, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric'),
                 dict(type='PrecisionRecallMetric', average='binary'),  # 二分类精确率/召回率
                 dict(type='F1Metric', average='binary')  # 二分类F1-score
                                        ]

# val_evaluator = [
#     # 整体性能
#     dict(type='AccMetric'),  # 总体准确率
#     dict(type='TopKAccMetric', k=3),  # Top-3准确率（适配多类）
#     # 单类表现
#     dict(type='PerClassAccuracy', num_classes=11),  # 每类准确率（替换为你的类别数）
#     dict(type='ConfusionMatrixMetric', num_classes=11, normalize='true'),  # 混淆矩阵
#     # 类别平衡表现
#     dict(type='F1Metric', average='macro'),  # 宏平均F1（关注少数类）
# ]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

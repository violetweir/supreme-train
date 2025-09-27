#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Configuration for StarNet with Channel Attention and Learnable Wavelets
model = dict(
    type='StarNet_CA_T2_LW',
    model_cfg=dict(
        img_size=224,
        dims=[40, 40, 80, 160, 320],
        depth=[1, 2, 8, 2],
        mlp_ratio=2,
        act_layer="GELU",
        drop_path_rate=0.0,
        learnable_wavelet=True,
        reduction=16
    ),
    num_classes=1000,
    pretrained=False,
    distillation=False,
)

augmentation = dict(
    train=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        mode='batch',
        cutmix_minmax=None,
        label_smoothing=0.1,
        num_classes=1000
    ),
    eval=dict(
        label_smoothing=0.1,
        num_classes=1000
    )
)

loss = dict(
    type='CrossEntropyLoss',
    label_smoothing=0.1,
    reduction='mean'
)

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1e-3,
    warmup_by_epoch=True
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=300
)

checkpoint_config = dict(
    interval=1
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

evaluation = dict(
    interval=1,
    metric='accuracy'
)

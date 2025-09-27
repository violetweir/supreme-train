_base_ = [
    '../_base_/models/mask_rcnn_efficientvit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    # pretrained=None,
    backbone=dict(
        _delete_=True,
        type='MobileMamba',
        img_size=192,
        in_chans=3,
        num_classes=80,
        stages=['s', 's', 's'],
        embed_dim=[144,272,368],
        global_ratio=[0.8, 0.7, 0.6],
        local_ratio=[0.2, 0.2, 0.3],
        depth=[1,2,2],
        kernels=[7, 5, 3],
        down_ops=[['subsample', 2], ['subsample', 2], ['']],
        distillation=False, drop_path=0.03, ssm_ratio=2, forward_type="v052d",
        sync_bn=False, out_indices=(1, 2, 3),
        pretrained='../../weights/MobileMamba_T2/mobilemamba_t2.pth',
        frozen_stages=-1, norm_eval=True, ),
        neck=dict(
              type='EfficientViTFPN',
              in_channels=[144,272,368],
              out_channels=256,
              start_level=0,
              num_outs=5,
              num_extra_trans_convs=2,
        ),
)

ratio = 1
bs_ratio = 2  # 0.0002 for 2 * 8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                   'relative_position_bias_table': dict(decay_mult=0.),
                                                   'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=0.1, norm_type=2), )

max_epochs = 12
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=max_epochs // 2,
        T_max=max_epochs // 2,
        end=max_epochs,
        by_epoch=True,
        eta_min=0)
]


train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 8),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader

_base_ = [
    '../_base_/models/mask_rcnn_efficientvit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]



model = dict(
    backbone=dict(
        _delete_=True,
        type='StarNet',
        img_size=224,
        in_chans=3,
        num_classes=80,
        dims=[32, 64, 128, 256],
        depth=[1,2,4,5], 
        mlp_ratio=2., 
        
        sync_bn=False, out_indices=(0,1, 2, 3),
        pretrained='/Data_8TB/lht/MobileMamba/model_weights/StarNet_MHSA_T2_DTW/net_E.pth',
        frozen_stages=-1, norm_eval=True, ),
    neck=dict(
        type='LSNetFPN',
        in_channels=[32,64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=5,
        num_extra_trans_convs=1,
        )
)



ratio = 1
bs_ratio = 2  # 0.0002 for 2 * 8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00005 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
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
    batch_size=1 * bs_ratio * ratio,
    num_workers=min(1 * bs_ratio * ratio, 8),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader

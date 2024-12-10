# optimizer - AdamW
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),  # Gradient Clipping 추가
    type='OptimWrapper',
    optimizer=dict(
        betas=(
            0.8,
            0.99,
        ), lr=5e-04, type='AdamW', weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))))


param_scheduler = [
    dict(
        type='LinearLR',  # Warm-up 스케줄러
        start_factor=0.001, 
        by_epoch=True,
        begin=0,
        end=5  # warm-up이 진행될 에폭 수 (예: 5 에폭 동안 warm-up 적용)
    ),
    dict(
        type='PolyLR', 
        eta_min=1e-2,
        power=0.8,
        begin=5, 
        end=300,
        by_epoch=True
    )
]

# training schedule for 500 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)  # validate every 10 epochs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# default hooks including early stopping
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # IterTimerHook을 유지
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_last=True,  max_keep_ckpts=1 ),  # save checkpoint every 10 epochs
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='target_class_dice',  # Metric to monitor
        patience=15,  # Number of epochs to wait for improvement
        min_delta=0.01,  # Minimum change to qualify as an improvement
        rule = 'greater'
    )
)

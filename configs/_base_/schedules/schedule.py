# optimizer - AdamW
optim_wrapper = dict(
    clip_grad=None,
    type='OptimWrapper',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))))

'''# optimizer - SGD+Momentum
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)'''

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-3,
        power=0.9,
        begin=0,
        end=500,
        by_epoch=True)
]

# training schedule for 500 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=500, val_interval=1)  # validate every 10 epochs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# default hooks including early stopping
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # IterTimerHook을 유지
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5),  # save checkpoint every 10 epochs
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='target_class_dice',  # Metric to monitor
        patience=35,  # Number of epochs to wait for improvement
        min_delta=0.01,  # Minimum change to qualify as an improvement
        rule = 'greater'
    )
)

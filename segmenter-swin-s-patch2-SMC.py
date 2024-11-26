_base_ = [
    'configs/_base_/models/segmenter_swin.py',
    'configs/_base_/datasets/SMC.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]

crop_size = (291, 80)
data_preprocessor = dict(size=crop_size)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    # backbone Swin-small-patch2-window7으로 수정
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        patch_size = 4
        window_size=5, # window-size 7->5 수정
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels = 768,
        embed_dims= 768,
        num_heads=12,
        num_layers=2,
        out_channels=2,
        dropout_ratio=0.0,
        num_classes=2,
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0),#Focal Loss
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(145, 40)),
)

train_dataloader = dict(batch_size=8) #batch-size
val_dataloader = dict(batch_size=1)


val_evaluator = dict(type='CustomDiceMetric', target_class_index=1,iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(
    format_only= True,
    keep_results=True,
    output_dir='',
    iou_metrics=['mIoU'],
    type='IoUMetric')


load_from = checkpoint
log_level = 'INFO'
log_processor = dict(by_epoch=True)

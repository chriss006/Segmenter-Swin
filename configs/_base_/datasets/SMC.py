# 데이터셋 타입과 경로를 수정
dataset_type = 'SMCDatasets'
data_root = '../SMC/datasets/'
test_data_root = '../SMC/datasets/test/'
test_img_dir = 'images'
test_mask_dir = 'labels'
train_img_dir = 'train/images/'
train_ann_dir = 'train/labels/'
val_img_dir = 'validation/images/'
val_ann_dir = 'validation/labels/'

crop_size = (80,291)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False ),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

# test 파이프라인 설정
test_pipeline = val_pipeline

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]


train_dataloader = dict(
    batch_size=8, 
    dataset=dict(
        type='SMCDatasets',
        data_root=data_root,
        data_prefix=dict(img_path=train_img_dir, seg_map_path=train_ann_dir),
        pipeline=train_pipeline,
        reduce_zero_label=False, 
        ignore_index=255 # 여기서 reduce_zero_label 설정
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='SMCDatasets',
        data_root=data_root,
        data_prefix=dict(img_path=val_img_dir, seg_map_path=val_ann_dir),
        pipeline=val_pipeline,
        reduce_zero_label=False,
        ignore_index=255 # 여기서 reduce_zero_label 설정
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

test_dataloader =  dict(
    batch_size=1,
    dataset=dict(
        type='SMCDatasets',
        data_root=test_data_root,
        data_prefix=dict(img_path=test_img_dir, seg_map_path=test_mask_dir),
        pipeline=test_pipeline,
        reduce_zero_label=False,
        ignore_index=255 # 여기서 reduce_zero_label 설정
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

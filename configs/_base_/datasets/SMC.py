# 데이터셋 타입과 경로를 수정
dataset_type = 'SMCDatasets'
data_root = '../SMC/datasets/'
test_data_root = '../SMC/datasets/test/'
test_img_dir = 'images'
test_mask_dir = 'labels'
img_dir = 'train/images'
ann_dir = 'train/labels'

crop_size = (291,80)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(291, 80), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(291, 80), cat_max_ratio=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

# validation 파이프라인 설정
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(291, 80), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),  
    dict(type='PackSegInputs'),
]

# test 파이프라인 설정
test_pipeline = val_pipeline

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.75, 1.0, 1.25]  # 주요 비율만 남김
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='PackSegInputs')]  # LoadAnnotations 제거
        ])
]

train_dataloader = dict(
    batch_size=8, 
    dataset=dict(
        type='SMCDatasets',
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=train_pipeline,
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
        ann_file='splits/train.txt', 
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
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=val_pipeline,
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
        ann_file='splits/val.txt',
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
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

import os

dataset_type = 'SMCDatasets'
data_root = '../SMC/dataset/'


# ann_file 경로
fold_id = 5
fold_train_split_txt = f'fold_splits/fold{fold_id}_train.txt'
fold_val_split_txt = f'fold_splits/fold{fold_id}_val.txt'
fold_test_split_txt = f'fold_splits/fold{fold_id}_test.txt'


# Fold-specific data_prefix 설정
img_dir = 'images/'
ann_dir = 'labels/'

crop_size = (80, 291)

# Train 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

# Validation 파이프라인
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

# Test 파이프라인 설정
test_pipeline = val_pipeline

# Train 데이터 로더
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=fold_train_split_txt,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=train_pipeline,
        reduce_zero_label=False,
        ignore_index=255,  # reduce_zero_label 설정
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

# Validation 데이터 로더
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=fold_val_split_txt,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=val_pipeline,
        reduce_zero_label=False,
        ignore_index=255,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

# Test 데이터 로더
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=fold_test_split_txt, 
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=test_pipeline,
        reduce_zero_label=False,
        ignore_index=255,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

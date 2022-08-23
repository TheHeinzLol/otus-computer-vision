dataset_type = 'AVADataset'
data_root = 'ava_mma/main_result/frames'
ann_file_train = 'ava_mma/main_result/dataset/HBVSMCGRG_TRAIN.csv'
ann_file_val = 'ava_mma/main_result/dataset/HBVSMCGRG_VAL.csv'
label_file = 'ava_mma/main_result/dataset/mma_action_list.pbtxt'
proposal_file_train = 'ava_mma/main_result/dataset/ava_proposals_train.pkl'
proposal_file_val = 'ava_mma/main_result/dataset/ava_proposals_val.pkl'
# делаем +1, 0 - резервный класс
num_classes = 8

train_timestamp_start = 0
train_timestamp_end = 208
val_timestamp_start = 0
val_timestamp_end = 54

clip_len = 32
frame_interval = 1
person_det_score_thr = 0.9
filename_tmpl = "img_{:04}.jpg"
num_max_proposals = 5
fps = 30

lr = 0.009375
total_epochs = 50

model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            temporal_pool_mode='max'),
        shared_head=dict(type='ACRNHead', in_channels=4608, out_channels=2304),
        bbox_head=dict(
            type='BBoxHeadAVA',
            dropout_ratio=0.5,
            in_channels=2304,
            num_classes=num_classes,
            multilabel=True)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=clip_len, frame_interval=frame_interval, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=None,
        proposal_file=proposal_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        person_det_score_thr=person_det_score_thr,
        data_prefix=data_root,
        num_classes=num_classes,
        timestamp_start=train_timestamp_start,
        timestamp_end=train_timestamp_end,
        filename_tmpl=filename_tmpl,
        num_max_proposals=num_max_proposals,
        fps=fps),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        proposal_file=proposal_file_val,
        exclude_file=None,
        pipeline=val_pipeline,
        label_file=label_file,
        person_det_score_thr=person_det_score_thr,
        data_prefix=data_root,
        num_classes=num_classes,
        timestamp_start=val_timestamp_start,
        timestamp_end=val_timestamp_end,
        filename_tmpl=filename_tmpl,
        num_max_proposals=num_max_proposals,
        fps=fps))
data['test'] = data['val']

# optimizer
#optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.00001)

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1)
total_epochs = total_epochs
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1)
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb'  # noqa: E501
load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'  # noqa: E501
resume_from = None
find_unused_parameters = False
gpu_ids = [0]
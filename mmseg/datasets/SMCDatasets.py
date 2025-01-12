from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os

# 각 클래스에 대한 팔레트(색상)을 정의
classes = ('background', 'meniscus')
palette = [[128, 0, 0], [0, 128, 0]]

@DATASETS.register_module()
class SMCDatasets(BaseSegDataset):
    METAINFO = dict(classes=classes, reduce_zero_label=False, palette=palette)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

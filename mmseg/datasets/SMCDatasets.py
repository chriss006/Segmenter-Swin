from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import numpy as np

# 각 클래스에 대한 팔레트(색상)을 정의
classes = ('background', 'meniscus')
palette = [[128, 0, 0], [0, 128, 0]]

@DATASETS.register_module()
class SMCDatasets(BaseSegDataset):
    # 클래스, 팔레트 정보에 대한 딕셔너리 METAINFO 생성
    METAINFO = dict(classes=classes, reduce_zero_label=False, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

    def _convert_label(self, label):
        """레이블 값을 0과 1로 매핑"""
        label = np.array(label, dtype=np.uint8)
        label[label == 128] = 1  # Meniscus 매핑
        return label
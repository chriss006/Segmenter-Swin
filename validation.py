# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import mmseg.datasets.SMCDatasets
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
import sys
from mmengine.logging import MMLogger

def main():
    # Configuration 파일 경로
    config_path = "segmenter-swin-s-patch2win7-SMC.py"
    work_dir = "../SMC/work_dirs/patch2/w7batch32lr5e-4/"
    
    # Config 파일 로드
    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir

    # Runner 초기화
    runner = Runner.from_cfg(cfg)

    # 검증만 수행
    print("Starting validation only...")
    runner.val_loop.run()

if __name__ == "__main__":
    main()

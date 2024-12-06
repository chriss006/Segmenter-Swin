# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS
from .iou_metric import IoUMetric

@METRICS.register_module()
class CustomDiceMetric(IoUMetric):
    """Custom Dice evaluation metric for a specific class.

    Args:
        target_class_index (int): Index of the class to be monitored.
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            include 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 target_class_index: int,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(ignore_index=ignore_index, iou_metrics=iou_metrics, nan_to_num=nan_to_num, beta=beta, collect_device=collect_device, output_dir=output_dir, format_only=format_only, prefix=prefix, **kwargs)
        self.target_class_index = target_class_index



    def compute_metrics(self, results: list) -> Dict[str, float]:
        results = tuple(zip(*results))
    
        # Sum up the areas for IoU/Dice calculation
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
    
        # Extract and compute loss values
        loss_dicts = results[4]

        # Calculate focal_loss, dice_loss, combined_loss 
        focal_loss_mean = np.mean([loss.get('loss_focal', 0.0) for loss in loss_dicts])
        dice_loss_mean = np.mean([loss.get('loss_dice', 0.0) for loss in loss_dicts])
        combined_loss_mean = focal_loss_mean + dice_loss_mean
        
        # Logging
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation Losses:')
        logger.info(f'  Focal Loss: {focal_loss_mean:.4f}')
        logger.info(f'  Dice Loss: {dice_loss_mean:.4f}')
        logger.info(f'  Combined Loss: {combined_loss_mean:.4f}')
    
        # Compute IoU/Dice metrics
        metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
    
        metrics['focal_loss'] = focal_loss_mean
        metrics['dice_loss'] = dice_loss_mean
        metrics['combined_loss'] = combined_loss_mean
    
        return metrics

    





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
        focal_loss_mean = np.mean([loss.get('loss_focal', 0.0) for loss in loss_dicts])
        dice_loss_mean = np.mean([loss.get('loss_dice', 0.0) for loss in loss_dicts])
        combined_loss_mean = focal_loss_mean + dice_loss_mean
    
        # Compute IoU/Dice metrics
        metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
    
        # Add loss values to metrics
        metrics['focal_loss'] = focal_loss_mean
        metrics['dice_loss'] = dice_loss_mean
        metrics['combined_loss'] = combined_loss_mean

        # Add target_class_dice
        dice_scores = metrics.get('Dice', [])  
        target_class_idx = getattr(self, 'target_class_index', 1)  
        target_class_dice = dice_scores[target_class_idx] if len(dice_scores) > target_class_idx else None
        if target_class_dice is not None:
            metrics['target_class_dice'] = target_class_dice

        # Generate class-wise table
        class_names = self.dataset_meta.get('classes', [])
        num_classes = len(class_names)
        class_table = PrettyTable()
    
        iou_scores = metrics.get('IoU', [0] * num_classes)
        acc_scores = metrics.get('Acc', [0] * num_classes)
        dice_scores = metrics.get('Dice', [0] * num_classes)

        class_table.add_column("Class", class_names)
        class_table.add_column("IoU", [ f'{score:.4f}' for score in iou_scores])
        class_table.add_column("Acc", [ f"{score:.4f}" for score in acc_scores])
        class_table.add_column("Dice", [ f"{score:.4f}" for score in dice_scores])
    
        # Log results
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation Losses:')
        logger.info(f'  Focal Loss: {focal_loss_mean:.4f}')
        logger.info(f'  Dice Loss: {dice_loss_mean:.4f}')
        logger.info(f'  Combined Loss: {combined_loss_mean:.4f}')
        if target_class_dice is not None:
            logger.info(f'  Target Class Dice: {target_class_dice:.4f}')
        print_log('Per class results:', logger)
        print_log('\n' + class_table.get_string(), logger=logger)
    
        # Return summary metrics with target_class_dice
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in metrics.items()
            if isinstance(ret_metric_value, (list, np.ndarray))  # Only numeric values
        })
        if target_class_dice is not None:
            ret_metrics_summary['target_class_dice'] = target_class_dice
        
        return ret_metrics_summary


    





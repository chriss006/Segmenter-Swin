from mmengine.registry import LOOPS
from mmengine.evaluator import Evaluator
from torch.utils.data import DataLoader
import torch
from mmengine.logging import MMLogger
from mmengine.utils import HistoryBuffer

@LOOPS.register_module()
class ValLoops(BaseLoop):
    def __init__(self, runner, dataloader: DataLoader, evaluator: Evaluator, fp16: bool = False):
        super().__init__(runner, dataloader)
        self.evaluator = evaluator
        self.fp16 = fp16
        self.val_loss = {}

    def run(self):
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self.val_loss.clear()

        logger = MMLogger.get_current_instance()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # Calculate metrics from evaluator
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        # Key indicator 설정 및 KeyError 방지
        key_indicator = self.runner.cfg.get('key_indicator', 'target_class_dice')  # 기본값 설정
        key_score = metrics.get(key_indicator, None)
        if key_score is None:
            logger.warning(f"Key indicator '{key_indicator}' not found in metrics.")
            key_score = 0.0
        
        # Process and log validation losses
        if self.val_loss:
            loss_dict = self._parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)
            logger.info(f"Validation Losses:")
            for loss_name, loss_value in loss_dict.items():
                logger.info(f"  {loss_name}: {loss_value:.4f}")

        # Call hooks after validation epoch
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)

        # Perform val_step
        with torch.cuda.amp.autocast(enabled=self.fp16):
            data_samples = self.runner.model.val_step(data_batch, loss=True)
        
        # Log or debug `loss_dict`
        for data_sample in data_samples:
            if 'loss_dict' not in data_sample:
                print(f"Warning: loss_dict is missing in data_sample {data_sample}")
            else:
                for loss_name, loss_value in data_sample['loss_dict'].items():
                    if loss_name not in self.val_loss:
                        self.val_loss[loss_name] = HistoryBuffer()
                    if isinstance(loss_value, torch.Tensor):
                        self.val_loss[loss_name].update(loss_value.item(), count=1)
    
        # Process results for evaluation
        self.evaluator.process(data_samples=data_samples, data_batch=data_batch)
        self.runner.call_hook('after_val_iter', batch_idx=idx, data_batch=data_batch, outputs=data_samples)



    def _update_losses(self, outputs, losses, stage):
        """Update loss values into `losses` dictionary."""
        if isinstance(outputs[-1], dict) and 'loss' in outputs[-1]:
            loss = outputs[-1]['loss']
            outputs = outputs[:-1]  # Remove loss dict from outputs
        else:
            print("Error: Outputs[-1] does not contain a 'loss' key.")
            print("Outputs[-1]:", outputs[-1])
            raise ValueError("Invalid outputs structure in _update_losses")

        # Update the losses dictionary
        for loss_name, loss_value in loss.items():
            full_loss_name = f"{stage}_{loss_name}"
            if full_loss_name not in losses:
                losses[full_loss_name] = HistoryBuffer()
            if isinstance(loss_value, torch.Tensor):
                losses[full_loss_name].update(loss_value.item(), count=1)
            elif isinstance(loss_value, list) and all(isinstance(lv, torch.Tensor) for lv in loss_value):
                for lv in loss_value:
                    losses[full_loss_name].update(lv.item(), count=1)
        return outputs, losses

    def _parse_losses(self, losses, stage):
        """Parse accumulated losses into a dictionary with averages."""
        all_loss = 0
        loss_dict = {}
        for loss_name, loss_value in losses.items():
            if loss_name.startswith(stage):
                avg_loss = loss_value.avg  # Use the average loss
                loss_dict[loss_name] = avg_loss
                if 'loss' in loss_name:
                    all_loss += avg_loss
        loss_dict[f'{stage}_loss'] = all_loss
        return loss_dict



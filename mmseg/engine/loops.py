from mmengine.registry import LOOPS
from mmengine.evaluator import Evaluator
from torch.utils.data import DataLoader
import torch

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
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        if self.val_loss:
            loss_dict = self._parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch, loss=True)
        outputs, self.val_loss = self._update_losses(outputs, self.val_loss, 'val')
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook('after_val_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)

    def _update_losses(self, outputs, losses, stage):
        if isinstance(outputs[-1], dict) and 'loss' in outputs[-1]:
            loss = outputs[-1]['loss']
            outputs = outputs[:-1]
        else:
            loss = {}
        for loss_name, loss_value in loss.items():
            full_loss_name = f"{stage}_{loss_name}"
            if full_loss_name not in losses:
                losses[full_loss_name] = HistoryBuffer()
            if isinstance(loss_value, torch.Tensor):
                losses[full_loss_name].update(loss_value.item())
            elif isinstance(loss_value, list) and all(isinstance(lv, torch.Tensor) for lv in loss_value):
                for lv in loss_value:
                    losses[full_loss_name].update(lv.item())
        return outputs, losses

    def _parse_losses(self, losses, stage):
        all_loss = 0
        loss_dict = {}
        for loss_name, loss_value in losses.items():
            if loss_name.startswith(stage):
                avg_loss = loss_value.mean()
                loss_dict[loss_name] = avg_loss
                if 'loss' in loss_name:
                    all_loss += avg_loss
        loss_dict[f'{stage}_loss'] = all_loss
        return loss_dict

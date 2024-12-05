from mmengine.hooks import Hook

class CustomEarlyStoppingHook(Hook):
    """Early stop if monitored metric drops significantly."""
    def __init__(self, monitor, drop_threshold, patience=5, strict=False):
        self.monitor = monitor
        self.drop_threshold = drop_threshold  # 감소 기준
        self.patience = patience
        self.strict = strict
        self.wait = 0
        self.prev_best = None

    def after_val_epoch(self, runner):
        current_score = runner.log_buffer.output.get(self.monitor)
        if current_score is None:
            if self.strict:
                raise ValueError(f"Metric {self.monitor} not found in metrics!")
            return

        # 이전 값이 설정되지 않았다면 현재 값 저장
        if self.prev_best is None:
            self.prev_best = current_score
            return

        # 감소량 확인
        if self.prev_best - current_score >= self.drop_threshold:
            runner.logger.info(
                f"Significant drop in {self.monitor}: {self.prev_best} -> {current_score}."
            )
            runner.should_stop = True
            return

        # 개선되지 않은 경우 patience 관리
        if current_score <= self.prev_best:
            self.wait += 1
            if self.wait >= self.patience:
                runner.logger.info(
                    f"No improvement in {self.monitor} for {self.patience} epochs. Stopping."
                )
                runner.should_stop = True
        else:
            # 개선된 경우 wait 초기화
            self.prev_best = current_score
            self.wait = 0

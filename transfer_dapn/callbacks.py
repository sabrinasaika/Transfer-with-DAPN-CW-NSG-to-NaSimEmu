"""WinRateCallback — rolling win rate logged to SB3's verbose table."""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class WinRateCallback(BaseCallback):
    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._results: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                win = bool(info.get("win", False))
                # Fallback: positive total return also counts as win
                if not win and ep.get("r", -999) > 0:
                    win = True
                self._results.append(1.0 if win else 0.0)
        if self._results:
            self.logger.record(
                "rollout/win_rate",
                sum(self._results) / len(self._results))
        return True

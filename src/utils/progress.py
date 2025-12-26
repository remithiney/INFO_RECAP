try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class Progress:
    def __init__(self):
        self._bar = None

    def set_stage(self, name, total):
        if tqdm is None:
            print(f"[{name}]")
            return
        if self._bar is None:
            self._bar = tqdm(total=total)
        else:
            self._bar.reset(total=total)
        self._bar.set_description(name)

    def update(self, n=1):
        if self._bar is not None:
            self._bar.update(n)

    def close(self):
        if self._bar is not None:
            self._bar.close()

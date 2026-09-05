from fields import BaseModel
from dataclasses import dataclass, field
import numpy as np

import matplotlib.pyplot as plt
from collections import deque

_SKIP = ("_grad_i", "_grad_j")


@dataclass
class ObserverConfig:
    percentiles: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]) # Percentiles to track (e.g. [0.0, 0.25, 0.5, 0.75, 1.0])
    hourly_keep: int = 30 * 24


class ObserverBuffer:
    def __init__(self, max_size, percentiles):
        self.max_size = max_size
        self.p = np.asarray(percentiles)
        self.buf, self.pbuf = [], []

    def _pctl(self, x):
        if x.ndim == 2:
            return np.percentile(x, self.p * 100)

        if x.ndim == 3:
            h, w, c = x.shape
            return np.percentile(x.reshape(-1, c), self.p * 100, 0).T  # (C, P)

        raise ValueError(x.ndim)

    def add(self, x):
        self.buf.append(x)
        self.pbuf.append(self._pctl(x))

        if len(self.buf) > self.max_size:
            self.buf.pop(0)
            self.pbuf.pop(0)

    def hourly(self):
        return self.pbuf[-1]

    def mean(self, n=None):
        x = self.pbuf if n is None else self.pbuf[-n:]
        return np.mean(np.stack(x), axis=0)


class Observer(BaseModel):
    info = {'name': 'observer', 'map_info': {}}

    def __init__(self, world):
        super().__init__(world)
        self.world = world
        self.cfg = ObserverConfig()
        self.stats = {}
        self._build()

    def _ok(self, x, k):
        return (
            x.ndim in (2, )
            and x.dtype in (np.float32, np.float64)
            and not k.endswith(_SKIP)
        )

    def _build(self):
        for k, f in self.world.area.maps.items():
            x = f()
            if self._ok(x, k):
                self.stats[k] = {
                    "buffer": ObserverBuffer(self.cfg.hourly_keep, self.cfg.percentiles),
                    "hourly": [],
                    "monthly": []
                }

    def step(self):
        for k, f in self.world.area.maps.items():
            x = f()
            if not self._ok(x, k):
                continue

            s = self.stats[k]
            s["buffer"].add(x)
            s["hourly"].append(s["buffer"].hourly())

            if self.world["time"].is_first_hour_of_month:
                s["monthly"].append(s["buffer"].mean(30 * 24))

    def plot(self, keys=None, n_columns=2, last_n=None):
        if keys is None:
            keys = list(self.stats.keys())

        rows = (len(keys) + n_columns - 1) // n_columns
        fig, axes = plt.subplots(rows, n_columns, figsize=(5 * n_columns, 4 * rows), layout='constrained')
        axes = np.atleast_1d(axes).ravel()

        for ax, k in zip(axes, keys):
            s = np.array(self.stats[k]['hourly'])  # IMPORTANT FIX

            if s.ndim != 2:
                raise ValueError(f"Expected (time, percentiles), got {s.shape}")

            if last_n is not None:
                if type(last_n) == int:
                    s = s[-last_n:]
                elif type(last_n) == tuple:
                    s = s[-last_n[0]:-last_n[1]]

            dt = self.world['time'].dt
            hours = np.arange(-len(s) + 1, 1) * dt
            days = hours / 24

            p = self.cfg.percentiles
            mid = len(p) // 2

            ax.plot(days, s[:, mid], label='median', color='blue')

            for i in range(mid):
                ax.fill_between(
                    days,
                    s[:, i],
                    s[:, -i-1],
                    alpha=0.2,
                    color='blue',
                    label=f'{int(p[i]*100)}-{int(p[-i-1]*100)}th pct' if i == 0 else None
                )

            ax.set_title(k)
            ax.set_xlabel('days ago')
            ax.set_ylabel('value')
            ax.grid(True)
            ax.legend()

        plt.show()
    
    def print(self, keys=None, frequency=100, last_n=None):
        last_date = self.world['time'].date
        if keys is None:
            keys = list(self.stats.keys())

        def fmt(x, width=12):
            return f"{x:{width}.6g}"

        p_labels = [f"p{int(100 * p):02d}" for p in self.cfg.percentiles]

        print("\nObserver Statistics")
        print("=" * 120)
        print(f"Current date       : {last_date}")
        print(f"Sampling frequency : every {frequency} observations")
        print(f"Percentiles        : {', '.join(p_labels)}")

        for k in keys:
            hourly = np.asarray(self.stats[k]["hourly"])

            if len(hourly) == 0:
                continue

            idx = np.arange(len(hourly))[::frequency]

            if last_n is not None:
                idx = idx[-last_n:]

            print(f"\n{k}")
            print("-" * 120)

            header = (
                f"{'step':>8} "
                f"{'day':>10} "
                + " ".join(f"{p:>12}" for p in p_labels)
            )
            print(header)

            for i in idx:
                vals = hourly[i]

                row = (
                    f"{i:8d} "
                    f"{i * self.world['time'].dt / 24:10.2f} "
                    + " ".join(fmt(v) for v in vals)
                )
                print(row)


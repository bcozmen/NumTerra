from pathlib import Path

import matplotlib.pyplot as plt

from fields import BaseModel
from .renderer import WorldRenderer


class WorldRendererGif(WorldRenderer):
    """Render each map to its own image file.

    ``save_path`` passed to :meth:`step` is treated as the output directory.
    A directory named after each map (lowercase) is created below it, and the
    current frame is saved as ``<map>/<date>-<hour>.jpg``.

    The rendering and colour-scale behaviour is shared with ``WorldRenderer``;
    only the figure layout and saving behaviour differ.
    """

    info = {
        "name": "renderer_gif",
        "map_info": {}
    }

    def __init__(
        self,
        world,
        keys: list[str] | None = None,
        cmaps: dict[str, str] | None = None,
        figsize: tuple[float, float] = (6.0, 5.0),
    ):
        BaseModel.__init__(self, world)
        self._extra_cmaps = cmaps or {}
        self.keys = keys if keys is not None else self._available_maps()
        if not self.keys:
            raise ValueError("No plottable maps found in world.area. "
                             "Run at least one model step first.")

        self._ims = {}
        self._cbars = {}
        self._titles = {}
        self._ax_map = {}
        self._contours = {}
        self._figures = {}
        self._axes = []

        # Build one independent figure per map. The inherited _build method
        # operates on self.fig and the current one-item self._axes list.
        for key in self.keys:
            self.fig, ax = plt.subplots(1, 1, figsize=figsize,
                                        layout="constrained")
            self._axes = [ax]
            self._figures[key] = self.fig
            self._build_keys = self.keys
            self.keys = [key]
            self._build()
            self.keys = self._build_keys

    def step(self, keys=None, save_path=None) -> None:
        """Redraw selected maps and optionally save one frame per map."""
        if keys is None:
            keys = self._ims

        if save_path is None:
            return

        time_model = self.world.models.get("time")
        date_hour = (time_model.tick.strftime("%Y-%m-%d-%H")
                     if time_model is not None else "frame")
        output_dir = Path(save_path) if save_path is not None else None

        for key in keys:
            if key not in self._ims:
                raise KeyError(f"Unknown renderer map: {key}")

            # Inherited rendering helpers use self.fig, so select the figure
            # belonging to this map before updating it.
            self.fig = self._figures[key]
            ax = self._ax_map[key]
            self._remove_contours(key)

            if key == "H":
                self._ims[key].set_data(self._height_rgba())
                self._titles[key].set_text(self._height_title())
            else:
                data, vmin, vmax = self._prepare_data(key)
                self._ims[key].set_data(data)
                self._ims[key].set_norm(self._make_norm(key, vmin, vmax))
                self._titles[key].set_text(self._panel_title(key, data))

            self._draw_contours(ax, key)
            if time_model is not None:
                self.fig.suptitle(str(time_model), fontsize=11)

            if output_dir is not None:
                map_dir = output_dir / key.lower()
                map_dir.mkdir(parents=True, exist_ok=True)
                self.fig.savefig(map_dir / f"{date_hour}.jpg", dpi=150)

            self._flush()
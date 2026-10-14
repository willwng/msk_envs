import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional
from cycler import cycler


@dataclass
class PlotConfig:
    # Size/layout
    num_vertical: int
    num_horizontal: int
    fig_size: tuple

    # Labels
    title: str
    x_label: str
    x_label_sub: str
    y_label: str

    # Ticks
    x_fmt: str
    x_sub_fmt: str
    y_fmt: str

    x_data: np.ndarray
    """ data for x axis (e.g., time) """
    x_data_sub: np.ndarray
    """ subscript for x axis (e.g., frame index) """


class SequencePlot:
    """ Represents plots on one page, tracking values over time """

    def __init__(self, cfg: PlotConfig):
        fig, axs = plt.subplots(cfg.num_vertical, cfg.num_horizontal,
                                figsize=cfg.fig_size)
        if cfg.num_vertical == 1 and cfg.num_horizontal == 1:
            axs = np.array([[axs]])

        fig.suptitle(cfg.title, fontsize=24, fontweight="bold")
        fig.supxlabel(cfg.x_label, fontsize=14, fontweight="bold")
        fig.supylabel(cfg.y_label, fontsize=14, fontweight="bold")

        self.fig = fig
        self.axs = axs.flatten()
        self.axs_per_height = cfg.num_vertical
        self.axs_per_width = cfg.num_horizontal

        self.x_data = cfg.x_data
        self.x_data_sub = cfg.x_data_sub
        self.x_label_sub = cfg.x_label_sub

        self.x_fmt = cfg.x_fmt
        self.x_sub_fmt = cfg.x_sub_fmt
        self.y_fmt = cfg.y_fmt

        self.plot_per_ax = np.zeros((cfg.num_horizontal, cfg.num_vertical))
        self.labels_per_ax = np.zeros((cfg.num_horizontal, cfg.num_vertical))
        self.y_ranges = np.zeros((cfg.num_horizontal, cfg.num_vertical, 2))
        self.y_ranges[:, :, 0] = np.inf
        self.y_ranges[:, :, 1] = -np.inf

        # Color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', ':', '-.']
        combined_cycle = cycler(linestyle=linestyles) * cycler(color=colors)
        for ax in self.axs:
            ax.set_prop_cycle(combined_cycle)

    def _get_idx(self, ax_idx_x: int, ax_idx_y: int):
        return ax_idx_y * self.axs_per_width + ax_idx_x

    def _get_x_y_idx(self, idx: int):
        ax_idx_x = idx % self.axs_per_width
        ax_idx_y = idx // self.axs_per_width
        return ax_idx_x, ax_idx_y

    def get_axes_at(self, ax_idx_x: int, ax_idx_y: int):
        ax_idx = self._get_idx(ax_idx_x, ax_idx_y)
        return self.axs[ax_idx]

    def _update_y_range(self, idx: int, y_data: np.ndarray):
        ax_idx_x, ax_idx_y = self._get_x_y_idx(idx)
        curr_min, curr_max = self.y_ranges[ax_idx_x, ax_idx_y, :]
        new_min, new_max = np.min(y_data), np.max(y_data)
        self.y_ranges[ax_idx_x, ax_idx_y, 0] = min(curr_min, new_min)
        self.y_ranges[ax_idx_x, ax_idx_y, 1] = max(curr_max, new_max)

    def _update_plot_counts(self, idx: int, y_data: np.ndarray, label: str):
        ax_idx_x, ax_idx_y = self._get_x_y_idx(idx)
        # Count plots per axis
        if len(y_data.shape) == 1:
            self.plot_per_ax[ax_idx_x, ax_idx_y] += 1
            if label != "":
                self.labels_per_ax[ax_idx_x, ax_idx_y] += 1
        else:
            self.plot_per_ax[ax_idx_x, ax_idx_y] += y_data.shape[1]
            if label != "":
                self.labels_per_ax[ax_idx_x, ax_idx_y] += y_data.shape[1]

    def add(
            self,
            idx: int,
            y_data: np.ndarray,
            label: str,
            alpha: float = 1.0,
            linestyle: Optional[str] = None,
            title: str = None
    ):
        ax = self.axs[idx]
        plot_kwargs = dict(label=label, alpha=alpha)
        if linestyle is not None:
            plot_kwargs["linestyle"] = linestyle
        ax.plot(self.x_data, y_data, **plot_kwargs)

        if title is not None:
            ax.set_title(title, fontsize=14, fontweight="bold")

        self._update_y_range(idx, y_data)
        self._update_plot_counts(idx, y_data, label)

    # scatter (x, y) data points
    def add_scatter(self, idx: int, x_data: np.ndarray, y_data: np.ndarray,
                    label: str, title: str = None, connect_line: bool = False,
                    labeled: bool = False):
        ax = self.axs[idx]

        if connect_line:
            ax.plot(x_data, y_data, label=label, marker="o", linestyle='-')
        else:
            ax.scatter(x_data, y_data, label=label)

        # Text labels for each point
        if labeled:
            for (x, y) in zip(x_data, y_data):
                ax.text(x, y, f"{y:.3f}", fontsize=3, va='bottom')

        if title is not None:
            ax.set_title(title, fontsize=14, fontweight="bold")

        self._update_y_range(idx, y_data)
        self._update_plot_counts(idx, y_data, label)

    def add_overlay(self, idx: int, y_data: list[float]):
        ax = self.axs[idx]
        ax2 = ax.twinx()
        ax2.plot(self.x_data, y_data, color='gray', linestyle='dashed', alpha=0.5)

    def add_hline(self, idx: int, y_value: float, label: str = "", color: str = 'r'):
        ax = self.axs[idx]
        ax.axhline(y=y_value, color=color, linestyle='--', label=label)

        # Update y range
        ax_idx_x, ax_idx_y = self._get_x_y_idx(idx)
        curr_min, curr_max = self.y_ranges[ax_idx_x, ax_idx_y, :]
        self.y_ranges[ax_idx_x, ax_idx_y, 0] = min(curr_min, y_value)
        self.y_ranges[ax_idx_x, ax_idx_y, 1] = max(curr_max, y_value)

    def add_hline_with_bars(self, idx: int, y_value: float, y_var: float, label: str = "", color: str = 'r'):
        ax = self.axs[idx]
        ax.axhline(y=y_value, color=color, linestyle='--')
        ax.fill_between(self.x_data, y_value - y_var, y_value + y_var, color=color, alpha=0.2, label=label)

        # Update y range
        ax_idx_x, ax_idx_y = self._get_x_y_idx(idx)
        curr_min, curr_max = self.y_ranges[ax_idx_x, ax_idx_y, :]
        self.y_ranges[ax_idx_x, ax_idx_y, 0] = min(curr_min, y_value)
        self.y_ranges[ax_idx_x, ax_idx_y, 1] = max(curr_max, y_value)

    def enforce_y_range(self, idx: int, y_min: float, y_max: float):
        ax_idx_x, ax_idx_y = self._get_x_y_idx(idx)
        curr_min, curr_max = self.y_ranges[ax_idx_x, ax_idx_y, :]
        self.y_ranges[ax_idx_x, ax_idx_y, 0] = min(curr_min, y_min)
        self.y_ranges[ax_idx_x, ax_idx_y, 1] = max(curr_max, y_max)

    def finish(self, pdf):
        x_start, x_end = np.min(self.x_data), np.max(self.x_data)
        x_start_s, x_end_s = np.min(self.x_data_sub), np.max(self.x_data_sub)

        last_ax_row = self.axs[-self.axs_per_width:]
        first_ax_col = self.axs[::self.axs_per_width]

        # Iterate over all axes
        for i in range(self.axs_per_width):
            for j in range(self.axs_per_height):
                ax = self.get_axes_at(i, j)

                # Nothing was plotted here
                if self.plot_per_ax[i, j] == 0:
                    continue

                # Limits
                ax.set_xlim(x_start, x_end)
                y_start, y_end = self.y_ranges[i, j, :]
                y_range = y_end - y_start
                delta = 0.1 * abs(y_range)
                ax.set_ylim(y_start - delta, y_end + delta)

                # Set ticks
                x_ticks = np.linspace(x_start, x_end, num=5)
                x_ticks_sub = np.linspace(x_start_s, x_end_s, num=5)
                y_ticks = np.linspace(y_start, y_end, num=5)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)

                # Set tick labels
                x_tick_labels = [(f"{x:{self.x_fmt}}\n" +
                                  f"({self.x_label_sub} {x_sub:{self.x_sub_fmt}})")
                                 for x, x_sub in zip(x_ticks, x_ticks_sub)]
                y_tick_labels = [f"{y:{self.y_fmt}}" for y in y_ticks]

                # Only the last row gets x tick labels
                if ax in last_ax_row:
                    ax.set_xticklabels(x_tick_labels, fontsize=12,
                                       fontweight="bold")
                else:
                    ax.set_xticklabels([""] * len(x_tick_labels))

                if ax in first_ax_col:
                    ax.set_yticklabels(y_tick_labels, fontsize=12,
                                       fontweight="bold")
                else:
                    ax.set_yticklabels([""] * len(y_tick_labels))

                # thicker lines
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(1.5)

                # Legend required
                if self.labels_per_ax[i, j] >= 1:
                    ax = self.get_axes_at(i, j)
                    if self.labels_per_ax[i, j] > 12:
                        fontsize = 4
                    elif self.labels_per_ax[i, j] > 5:
                        fontsize = 6
                    else:
                        fontsize = 12
                    ax.legend(loc="upper right", frameon=False, prop={'weight': 'bold', 'size': fontsize})

                pass
            pass

        pdf.savefig(self.fig)
        plt.close(self.fig)

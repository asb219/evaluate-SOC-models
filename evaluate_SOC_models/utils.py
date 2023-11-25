from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger


def _save_show_close(plot_function):

    def new_plot_function(*args, save='', dpi=500, show=True, close=True, **kwargs):
        fig, ax = plot_function(*args, **kwargs)
        if save or show:
            plt.tight_layout()
        if save:
            path = Path(save).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=dpi)
            logger.success(f'Saved figure to "{path}".')
        if show:
            plt.show()
        elif close:
            plt.close()
        return fig, ax

    return new_plot_function

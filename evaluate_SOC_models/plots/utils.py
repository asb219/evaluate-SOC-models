from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger


def _save_show_close(plot_function):

    def new_plot_function(
        *args, save='', show=True, close=True, save_kwargs={}, **kwargs
    ):
        return_value = plot_function(*args, **kwargs)
        if save or show:
            plt.tight_layout()
        if save:
            path = Path(save).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, **save_kwargs)
            logger.debug(f'Saved figure to "{path}".')
        if show:
            plt.show()
        elif close:
            plt.close()
        return return_value

    return new_plot_function

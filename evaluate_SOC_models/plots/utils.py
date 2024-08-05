"""
Define useful decorator for plot functions, savefig extension with PIL.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import PIL
from loguru import logger


def _save_show_close(plot_function):

    def new_plot_function(*args, save=[], show=True, close=True,
            save_kwargs={}, return_save_paths=False, **kwargs):
        if not isinstance(save, list):
            save = [save]
        if not isinstance(save_kwargs, list):
            save_kwargs = [save_kwargs] * len(save)
        return_value = plot_function(*args, **kwargs)
        if save or show:
            plt.tight_layout()
        paths = [_savefig(s, **s_kw) for s, s_kw in zip(save, save_kwargs)]
        if show:
            plt.show()
        elif close:
            plt.close()
        return (return_value, paths) if return_save_paths else return_value

    return new_plot_function


def _savefig(save, *, remove_alpha_channel=False, P_image=False, P_colors=256,
        **save_kwargs):
    path = Path(save).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **save_kwargs)
    if remove_alpha_channel or P_image:
        img = PIL.Image.open(path)
        if remove_alpha_channel:
            img = img.convert('RGB') # convert from 'RGBA'
        if P_image:
            img = img.convert('P', palette=PIL.Image.ADAPTIVE, colors=P_colors)
        img.save(path, optimize=True)
    logger.debug(f'Saved figure to "{path}".')
    return path

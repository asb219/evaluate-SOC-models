import functools
from pathlib import Path
from loguru import logger


def yes_no_question(question, default=None):
    """Ask yes/no question and record user input.
    
    Parameters
    ----------
    question : str
        Question to ask user
    default : bool, optional
        Default answer if user input is left blank
    
    Returns
    -------
    answer : bool
        User's answer to yes/no question
    """

    if default is None:
        yn = ' (y/n) '
    elif default is True:
        yn = ' ([y]/n) '
    elif default is False:
        yn = ' (y/[n]) '
    else:
        raise TypeError('Argument `default` must be bool or None'
            f" but is of type '{type(default)}'")

    while True:
        answer = input(question + yn).strip().lower()
        if not answer:
            if default is not None:
                return default
        elif answer in ('y','yes'):
            return True
        elif answer in ('n','no'):
            return False


def clean_path(path):
    "Resolve `path`, but do not read the link if `path` is a symlink."
    # do not resolve directly in case file is a symlink
    path = Path(path).expanduser()
    if path.is_symlink():
        path = path.parent.resolve() / path.name
    else:
        path = path.resolve()
    return path


def resolve_path(path):
    "Resolve `path`. If `path` is a symlink, that is resolved too."
    return Path(path).expanduser().resolve()


def _rm_deco(function):
    @functools.wraps(function)
    def new_function(path, *args, **kwargs):
        path = clean_path(path)
        if not path.exists():
            raise FileNotFoundError(f'Cannot remove non-existent path: {path}')
        return function(path, *args, **kwargs)
    return new_function


@_rm_deco
def remove_path(path, ask=True, recursive=False):
    """Remove file or symlink `path`.
    If `recursive=True`, `path` can also be a directory.
    """

    if path.is_file():
        return remove_file(path, ask=ask)

    if path.is_symlink():
        return remove_symlink(path, ask=ask)

    if path.is_dir():
        if not recursive:
            raise ValueError(
                f'Cannot remove directory with recursive=False: {path}')
        if ask and not yes_no_question(f'Go into directory "{path}"?'):
            print(f'Do not go into directory.')
            return False
        else:
            return remove_tree(path, ask=ask)

    raise ValueError('Can only remove files, symlinks, and directories. '
        f'Cannot remove path: {path}')


@_rm_deco
def remove_file(path, ask=True):
    """Remove regular file."""
    if not path.is_file():
        raise ValueError(f'Path is not a file: {path}')

    if ask and not yes_no_question(f'Remove file "{path}"?'):
        print('Do not remove.')
        return False

    path.unlink()
    logger.info(f'Removed file: {path}')
    return True


@_rm_deco
def remove_symlink(path, ask=True):
    """Remove symbolic link."""
    if not path.is_symlink():
        raise ValueError(f'Path is not a symlink: {path}')

    if ask and not yes_no_question(f'Remove symlink "{path}"?'):
        print('Do not remove.')
        return False

    path.unlink()
    logger.info(f'Removed symlink: {path}')
    return True


@_rm_deco
def remove_directory(path, ask=True):
    """Remove empty directory."""
    if not path.is_dir() or any(path.iterdir()):
        raise ValueError(f'Path is not an empty directory: {path}')

    if ask and not yes_no_question(f'Remove empty directory "{path}"?'):
        print('Do not remove.')
        return False

    path.rmdir()
    logger.info(f'Removed empty directory: {path}')
    return True


@_rm_deco
def remove_tree(path, ask=True):
    """Recursively remove directory tree."""
    if not path.is_dir():
        raise ValueError(f'Path is not a directory: {path}')

    removed = {child: remove_path(child, ask=ask, recursive=True)
                for child in path.iterdir()}

    if all(removed.values()):
        removed[path] = remove_directory(path, ask=ask)

    return removed


def remove_pattern(parentdir, pattern, ask=True, recursive=False):
    """Remove files matching `pattern` in directory `parentdir`.
    If `recursive=True`, also remove directories.
    Note: Ensures that every object to remove is inside `parentdir`.
    """
    parentdir = resolve_path(parentdir)
    if not parentdir.is_dir():
        raise ValueError(f'`parentdir` is not a directory: {parentdir}')
    if unsafe_pattern(pattern):
        logger.warning(f'Pattern "{pattern}" is unsafe for removal')

    removed = {}

    for path in parentdir.glob(pattern):
        if not path_in_parentdir(path, parentdir):
            logger.error(f'Child "{path}" is not in parentdir "{parentdir}"')
            logger.info(f'Skip removal of "{path}"')
        else:
            removed[path] = remove_path(path, ask, recursive)

    return removed


def unsafe_pattern(pattern):
    """`True` if filename pattern is deemed unsafe, `False` otherwise.
    Removing an unsafe pattern could result in removing files outside
    of the intended directory.
    """
    return (pattern == '..') or ('../' in pattern) or pattern.endswith('/..')


def path_in_parentdir(path, parentdir):
    """`True` if `path` is inside of `parentdir`, `False` otherwise."""
    path = clean_path(path)
    parentdir = resolve_path(parentdir)
    return parentdir in path.parents

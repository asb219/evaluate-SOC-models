"""
Basic file types
"""

from pathlib import Path
import shutil
from zipfile import ZipFile
import pandas as pd
from loguru import logger

from data_manager.utils import yes_no_question


__all__ = ['File', 'FileGroup', 'Archive']


class File(object):
    """Generic file."""

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str or os.PathLike
            Path of the file
        """
        self.path = path

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = self._assert_good_filename(filename)
        self._path = self._savedir / self._filename

    @property
    def savedir(self) -> Path:
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        self._savedir = Path(savedir).expanduser().resolve()
        self._path = self._savedir / self._filename

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path):
        path = Path(path).expanduser().resolve()
        self._path = path
        self._savedir = path.parent
        self._filename = self._assert_good_filename(path.name)

    def _assert_good_filename(self, filename):
        filename = str(filename).strip()
        assert filename and ('/' not in filename) and (filename[0] != '.')
        return filename

    def remove(self, ask=True, missing_okay=False, **kwargs):
        """Remove file. Return `True` if removed, `False` otherwise."""
        #logger.debug(f'Removing file: {self}')
        if not self.exists():
            if missing_okay:
                return False
            #logger.debug(f'File does not exist, cannot remove file: {self}')
            raise FileNotFoundError(f'Cannot remove nonexistent file: {self}')
        if ask and not yes_no_question(f'Remove file "{self}"?'):
            print('Do not remove.')
            logger.debug(f'UIN: Do not remove file: {self}')
            return False
        self._remove(**kwargs)
        logger.debug(f'Removed file: {self}')
        return True

    def _remove(self, **kwargs):
        self.path.unlink()

    def exists(self):
        """Return `True` if file exists, `False` otherwise."""
        return self.path.exists()

    def claim(self):
        """Assert that file exists and return file object.
        
        Raises
        ------
        FileNotFoundError
            File does not exist
        """
        if not self.exists():
            raise FileNotFoundError(f'No such file or directory: {self}')
        return self

    def __repr__(self):
        return self.__class__.__name__ + f'("{self.path}")'

    def __str__(self):
        return str(self.path)


class FileGroup(object):
    """Group of :py:class:`File` objects."""

    def __init__(self, files, name=None):
        """
        Parameters
        ----------
        files: castable to pandas.Series
            Files in the file group
        name: str, optional
            Name of the file group
        """
        self._file_type_list = self._get_file_type_list()
        self.files = files
        self.name = name #: Name of the file group

    @property
    def files(self) -> "pandas.Series[File]":
        """Files belonging to the file group"""
        return self._files

    @files.setter
    def files(self, files):
        files = pd.Series(files, dtype='O')
        for file_type in self.file_type_list:
            if not all(isinstance(f, file_type) for f in files):
                raise TypeError(f'All files must be instances of {file_type}')
        self._files = files

    @property
    def file_type_list(self) -> list[type]:
        """List of types required for all the files in the group"""
        return self._file_type_list

    @classmethod
    def _get_file_type_list(cls):
        return [File]

    @property
    def paths(self) -> "pandas.Series[pathlib.Path]":
        """Paths of the files in the group"""
        return self.files.apply(lambda f: f.path)

    def remove(self, ask=True, missing_okay=True, **kwargs):
        """Remove all files in the group."""
        #logger.debug(f'Removing file group: {self}')
        removed = self.files.apply(lambda f: f.remove(ask, True, **kwargs))
        n = removed.sum()
        if not missing_okay and n==0:
            raise Exception
        logger.debug(f'Removed {n} files from file group: {self}')
        return removed

    def all_exist(self):
        """Return `True` if all files in the group exist, `False` otherwise."""
        return all(f.exists() for f in self.files)

    def claim(self):
        """Claim every file in the group and return file group object."""
        for f in self.files:
            f.claim()
        return self

    def items(self):
        """Generate (key, value) pairs for the files in the group."""
        return self.files.items()

    def __getitem__(self, item):
        return self.files[item]

    def __iter__(self):
        return iter(self.files)

    def __repr__(self):
        name = 'name=None' if self.name is None else f'name="{self.name}"'
        nfiles = f'nfiles={len(self.files)}'
        string = f'({name}, {nfiles})'
        return self.__class__.__name__ + string

    def __str__(self):
        string = '()' if self.name is None else f'("{self.name}")'
        return self.__class__.__name__ + string



class Archive(File):
    """Archive file manageable by python's :py:mod:`zipfile` package."""

    def namelist(self):
        """List names of files in archive."""
        with self.get_zip_file() as zip_file:
            return zip_file.namelist()

    def get_zip_file(self, mode='r', **kwargs):
        """Return an open :py:class:`zipfile.ZipFile` object,
        in read-only mode by default.
        """
        return ZipFile(self.claim().path, mode, **kwargs)

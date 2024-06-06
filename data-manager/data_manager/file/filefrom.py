# Copyright (c) 2024 Alexander S. Brunmayr. Subject to the MIT license.

"""
Fetched files
"""

import shutil
import os
import re
import urllib
from tqdm import tqdm

from loguru import logger

from .utils import yes_no_question
from .basefile import File, FileGroup, Archive


__all__ = [
    'FileFrom',
    'FileGroupFrom',
    'FileFromArchive',
    'FileGroupFromArchive',
    'FileFromDownload',
    'FileGroupFromDownload',
    'FileFromURL'
]


class FileFrom(File):
    """Base class for a file fetched from a source.
    
    Subclasses should overwrite the :py:meth:`fetch` method.
    """

    def claim(self):
        """Call :py:meth:`fetch` if file does not exist,
        then return file object."""
        if not self.exists():
            self.fetch()
        return self

    def fetch(self):
        """Fetch the file. Not implemented in the base class."""
        raise NotImplementedError

    def _remove_tmp_file(self, ask=False):
        """Remove temporary file(s) created while fetching the file."""
        logger.info(f'Removing temporary file(s) of {self}')
        pattern = self._tmp_filename
        results = [] if pattern is None else list(self.savedir.glob(pattern))
        if not results:
            logger.info('No temporary files to remove.')
            return
        for path in results:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    logger.info(f'Removed temporary file tree: {path}')
                else:
                    path.unlink()
                    logger.info(f'Removed temporary file: {path}')
            except:
                logger.error(f'Could not remove temporary file: {path}')
        logger.info(f'Done removing temporary file(s) of {self}')

    @property
    def _tmp_filename(self):
        return None


class FileGroupFrom(FileGroup):
    """Group of :py:class:`FileFrom` objects."""

    @classmethod
    def _get_file_type_list(cls):
        return super()._get_file_type_list() + [FileFrom]

    def fetch(self):
        """Fetch all the files in the group."""
        for f in self.files:
            f.fetch()



class FileFromArchive(FileFrom):
    """
    File extracted from an archive with :py:mod:`zipfile`.
    
    **Note**: strict typing for `archive` and `archived_filename`
    arguments.
    """

    def __init__(self, path, archive, archived_filename, **kwargs):
        """
        archive : Archive
            Archive containing the file
        archived_filename : str or re.Pattern
            Name of file in archive
        """
        if not isinstance(archive, Archive):
            raise TypeError('`archive` must be instance of Archive'
                f' but is of type {type(archive)}')
        if not isinstance(archived_filename, (str, re.Pattern)):
            raise TypeError('`archived_filename` must be str or re.Pattern'
                f' but is of type {type(archived_filename)}')

        super().__init__(path, **kwargs)
        self.archive = archive
        self.archived_filename = archived_filename

    __init__.__doc__ = FileFrom.__init__.__doc__ + __init__.__doc__


    def fetch(self):
        """Call :py:meth:`extract`."""
        self.extract(force=False, ask=False)


    def extract(self, force=False, ask=False, **kwargs):
        """Extract file from archive.
        
        Parameters
        ----------
        force : bool, default False
            If file already exists, remove and extract again
        ask : bool, default False
            Ask user for permission to extract or remove file
        
        Returns
        -------
        extract_path : str
            Path of extracted file, `None` if not extracted
        
        Other parameters
        ----------------
        **kwargs
            Keyword arguments passed to :py:meth:`_extract`
        """
        logger.debug(f'Extracting file: {self}')

        if ask and not yes_no_question(f'Extract "{self.filename}" from archive?'):
            print('Do not extract from archive.')
            logger.debug(f'UIN: Do not extract file: {self}')
            return None

        if self.exists():
            logger.info(f'File already exists: {self}')
            if not force or not self.remove(ask=ask):
                logger.info(f'Do not extract file: {self}')
                return None

        self.archive.claim()
        self.savedir.mkdir(parents=True, exist_ok=True)

        assert not self.exists()
        logger.info(f'Extracting "{self.filename}"...')

        try:
            extract_path = self._extract(**kwargs)
        except (KeyboardInterrupt, Exception):
            logger.exception(f'Extract failed or stopped: {self}')
            self._remove_tmp_file(ask=ask)
            raise
        else:
            if extract_path is None:
                logger.debug('Saved to unknown path.')
            else:
                logger.debug('Saved to "{extract_path}".')

        try:
            assert self.exists()
        except AssertionError:
            logger.exception(f'Extraction failed. File not found: {self}')
            raise
        else:
            logger.success(f'Extraction successful. File found: {self}')

        return extract_path


    def _extract(self, **kwargs):
        filename = self.archived_filename
        rename = self.filename
        path = str(self.savedir)
        return self.archive.extract_file(filename, path, rename, **kwargs)


    @property
    def _tmp_filename(self):
        return self.filename


class FileGroupFromArchive(FileGroupFrom):
    """Group of files extracted from archive"""

    @classmethod
    def _get_file_type_list(cls):
        return super()._get_file_type_list() + [FileFromArchive]

    def extract(self, force=False, ask=False, **kwargs):
        """Extract all files in the group from their respective archives."""
        return self.files.apply(lambda f: f.extract(force, ask, **kwargs))



class FileFromDownload(FileFrom):
    """Base class for a downloaded file.
    
    Subclasses should redefine the hidden :py:meth:`_download` method.
    """


    def fetch(self):
        """Call :py:meth:`download`."""
        self.download(force=False, ask=False)


    def download(self, force=False, ask=False, **kwargs):
        """Download file.
        
        Parameters
        ----------
        force : bool, default False
            If file already exists, remove and download again
        ask : bool, default False
            Ask user for permission to download or remove file
        
        Returns
        -------
        DL_path : str
            Path of downloaded file, `None` if not downloaded
        
        Other parameters
        ----------------
        **kwargs
            Keyword arguments passed to :py:meth:`_download`
        """
        logger.debug(f'Downloading file: {self}')

        if ask and not yes_no_question(f'Download "{self.filename}"?'):
            print('Do not download.')
            logger.debug(f'UIN: Do not download file: {self}')
            return None

        if self.exists():
            logger.info(f'File already exists: {self}')
            if not force or not self.remove(ask=ask):
                logger.info(f'Do not download file: {self}')
                return None

        self.savedir.mkdir(parents=True, exist_ok=True)

        assert not self.exists()
        logger.info(f'Downloading "{self.filename}"...')

        try:
            DL_path = self._download()
        except (KeyboardInterrupt, Exception):
            logger.exception(f'Download failed or stopped: {self}')
            self._remove_tmp_file(ask=ask)
            raise
        else:
            logger.debug(f'Saved to "{DL_path}".')

        try:
            assert self.exists()
        except AssertionError:
            logger.exception(f'Download failed. File not found: {self}')
            raise
        else:
            logger.success(f'Download successful. File found: {self}')

        return DL_path


    def _download(self, **kwargs):
        raise NotImplementedError


class FileGroupFromDownload(FileGroupFrom):
    """Group of downloaded files."""

    @classmethod
    def _get_file_type_list(cls):
        return super()._get_file_type_list() + [FileFromDownload]

    def download(self, force=False, ask=False, **kwargs):
        """Download all files in the group."""
        return self.files.apply(lambda f: f.download(force, ask, **kwargs))



class FileFromURL(FileFromDownload):
    """File downloaded with :py:mod:`urllib`.
    Uses :py:mod:`tqdm` progress bar."""

    def __init__(self, path, url, **kwargs):
        """
        url : str
            Download link
        """
        super().__init__(path, **kwargs)
        self.url = url

    __init__.__doc__ = FileFromDownload.__init__.__doc__ + __init__.__doc__


    def _download(self, **kwargs):
        with self._ProgressBar(**kwargs) as progress_bar:
            filename, headers = urllib.request.urlretrieve(
                url=self.url, filename=str(self.path),
                reporthook=progress_bar.update_to, data=None
            )
        return filename

    @property
    def _tmp_filename(self):
        return self.filename


    class _ProgressBar(tqdm):
        """Progress bar for downloads. Its :py:meth:`update_to` method is
        passed as a `reporthook` to :py:func:`urllib.request.urlretrieve`.
        
        **Note**: The implementation of this :py:class:`tqdm<tqdm.tqdm>`
        subclass is taken from
        https://github.com/tqdm/tqdm#hooks-and-callbacks
        """

        def __init__(self, *args, **kwargs):
            kwargs.setdefault('unit', 'B')
            kwargs.setdefault('unit_scale', True)
            kwargs.setdefault('unit_divisor', 1024)
            kwargs.setdefault('miniters', 1)
            super().__init__(*args, **kwargs)

        def update_to(self, b=1, bsize=1, tsize=None):
            """
            Wrapper for :py:meth:`tqdm.update<tqdm.tqdm.update>`.
            
            Parameters
            ----------
            b  : int, default 1
                Number of blocks transferred so far
            bsize  : int, default 1
                Size of each block (in tqdm units)
            tsize  : int, optional
                Total size (in tqdm units)
            """
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

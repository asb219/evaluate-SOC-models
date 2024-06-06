# Copyright (c) 2024 Alexander S. Brunmayr. Subject to the MIT license.

"""
Data files
"""

import pandas as pd
import xarray as xr
import rioxarray

from loguru import logger

from .basefile import File, FileGroup


__all__ = [
    'DataFile',
    'DataFileGroup',
    'PandasPickleFile',
    'PandasCSVFile',
    'PandasExcelFile',
    'XarrayNetCDFFile',
    'XarrayNetCDFFileGroup',
    'XarrayGeoTIFFile'
]


class DataFile(File):
    """Base class for a file containing data.
    
    Subclasses should overwrite the hidden :py:meth:`_read` and
    :py:meth:`_write` methods.
    """

    def __init__(self, path, *args, readonly=False, **kwargs):
        """
        readonly : bool, default False
            Make this file read-only
        """
        super().__init__(path, *args, **kwargs)
        self.readonly : bool = readonly #: If `True`, disable :py:meth:`write`

    __init__.__doc__ = File.__init__.__doc__ + __init__.__doc__

    def read(self, *args, **kwargs):
        """Read data from file."""
        return self.claim()._read(*args, **kwargs)

    def write(self, *args, **kwargs):
        """Write data to file."""
        if self.readonly:
            raise Exception(f'Cannot write to read-only file: {self}')
        self.savedir.mkdir(parents=True, exist_ok=True)
        try:
            self._write(*args, **kwargs)
        except (KeyboardInterrupt, Exception):
            logger.error(f'Write failed or stopped: {self}')
            self.remove(ask=False, missing_okay=True)
            raise
        else:
            logger.debug(f'Wrote to file: {self}')

    def _read(self, *args, **kwargs):
        raise NotImplementedError

    def _write(self, *args, **kwargs):
        raise NotImplementedError


class DataFileGroup(FileGroup):
    """Group of :py:class:`DataFile` objects."""

    def __init__(self, files, name=None, readonly=False, **kwargs):
        """
        readonly : bool, default False
            Make this file group read-only
        """
        super().__init__(files, name, **kwargs)
        self.readonly : bool = readonly #: If `True`, disable :py:meth:`write`

    __init__.__doc__ = FileGroup.__init__.__doc__ + __init__.__doc__

    @classmethod
    def _get_file_type_list(cls):
        return super()._get_file_type_list() + [DataFile]

    def read(self, *args, **kwargs):
        """Read data from the file group."""
        return self.claim()._read(*args, **kwargs)

    def write(self, *args, **kwargs):
        """Write data to the file group."""
        if self.readonly:
            raise Exception(f'File group is not writable: {self}')
        for f in self.files:
            f.savedir.mkdir(parents=True, exist_ok=True)
        try:
            self._write(*args, **kwargs)
        except (KeyboardInterrupt, Exception):
            logger.error(f'Write failed or stopped: {self}')
            self.remove(ask=False, missing_okay=True)
            raise
        else:
            logger.debug(f'Wrote to file group: {self}')

    def _read(self, *args, **kwargs):
        raise NotImplementedError

    def _write(self, *args, **kwargs):
        raise NotImplementedError



class PandasPickleFile(DataFile):
    """Pickle file (may be compressed) managed by :py:mod:`pandas`."""

    def _read(self, **kwargs):
        return pd.read_pickle(self.path, **kwargs)

    def _write(self, pandas_object, **kwargs):
        return pandas_object.to_pickle(self.path, **kwargs)


class PandasCSVFile(DataFile):
    """CSV file managed by :py:mod:`pandas`."""

    def _read(self, **kwargs):
        return pd.read_csv(self.path, **kwargs)

    def _write(self, pandas_object, **kwargs):
        return pandas_object.to_csv(self.path, **kwargs)


class PandasExcelFile(DataFile):
    """Excel file managed by :py:mod:`pandas`.
    Uses :py:mod:`openpyxl` for `xlsx` files and :py:mod:`xlrd`
    for `xls` files.
    """

    def __init__(self, path, *args, sheet_name=None, **kwargs):
        """
        sheet_name : str or int
            Name or index of default sheet in Excel file
        """
        super().__init__(path, *args, **kwargs)
        self.sheet_name = sheet_name

    __init__.__doc__ = DataFile.__init__.__doc__ + __init__.__doc__

    def _read(self, **kwargs):
        kwargs.setdefault('sheet_name', self.sheet_name)
        return pd.read_excel(self.path, **kwargs)

    def _write(self, pandas_object, *, mode='a', if_sheet_exists='replace',
                **kwargs):
        if not self.exists():
            mode = 'w'
            if_sheet_exists = None
        kwargs.setdefault('sheet_name', self.sheet_name)
        with pd.ExcelWriter(str(self.path), mode=mode,
                            if_sheet_exists=if_sheet_exists) as writer:
            return pandas_object.to_excel(writer, **kwargs)


class XarrayNetCDFFile(DataFile):
    """NetCDF file managed by :py:mod:`xarray`."""

    def _read(self, **kwargs):
        with xr.open_dataset(self.path, **kwargs) as ds:
            return ds # xarray DataSet

    def _write(self, xarray_object, **kwargs):
        return xarray_object.to_netcdf(self.path, **kwargs)


class XarrayNetCDFFileGroup(DataFileGroup):
    """NetCDF file group managed by :py:mod:`xarray`.
    Handles datasets spread over multiple NetCDF files."""

    @classmethod
    def _get_file_type_list(cls):
        return super()._get_file_type_list() + [XarrayNetCDFFile]

    def _read(self, **kwargs):
        with xr.open_mfdataset(self.paths, **kwargs) as ds:
            return ds # xarray DataSet


class XarrayGeoTIFFile(DataFile):
    """GeoTIF file opened with :py:func:`rioxarray.open_rasterio`."""

    def _read(self, **kwargs):
        with rioxarray.open_rasterio(self.path, **kwargs) as data:
            return data # xarray DataArray or DataSet

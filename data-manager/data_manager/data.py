# Copyright (c) 2024 Alexander S. Brunmayr. Subject to the MIT license.

from fnmatch import fnmatch
from pathlib import Path

from loguru import logger

from .file import PandasPickleFile, PandasCSVFile, PandasExcelFile
from .file import FileGroup


__all__ = [
    'Data',
    'DatasetNotFoundError'
]


class DatasetNotFoundError(Exception):
    """Name of dataset is not listed in :py:attr:`Data.datasets`."""

    def __init__(self, dataset, cls=None):
        message = f"No dataset named '{dataset}'"
        if isinstance(cls, Data):
            message += f' in {cls.__class__.__name__}'
        elif issubclass(cls, Data):
            message += f' in {cls.__name__}'
        super().__init__(message)



class Data:
    """
    Hosts a collection of datasets as :py:class:`pandas.DataFrame`
    or :py:class:`pandas.Series`.
    It handles data preprocessing and also provides shortcuts
    to output the datasets as Pickle, CSV, and Excel files.
    It is intended as a base class, where subclasses should
    overwrite the :py:attr:`datasets` class attribute and provide
    a :py:meth:`_process_\<dataset\>` method for each dataset.
    
    Datasets can be accessed by name via the :py:meth:`get` method
    or through indexing. Datasets may also be accessed as attributes.
    
    Parameters
    ----------
    savedir : str or os.PathLike
        Directory where datasets are written to file
    name : str
        Name of data
    description : str, optional
        Description of data
    
    Other parameters
    ----------------
    save_pkl : bool, default False
        Automatically write datasets to `.pkl.gz` file
    save_csv : bool, default False
        Automatically write datasets to `.csv` file
    save_xlsx : bool, default False
        Automatically write datasets to `.xlsx` file
    """

    datasets = []
    "Names of available datasets (class attribute)"

    def __init__(self, savedir, name, description=None,
            *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = Path(savedir).expanduser().resolve()
        name = str(name)

        self._savedir = savedir
        self._name = name
        self._description = description

        self._auto_save = {
            'pickle': {d: save_pkl for d in self.datasets},
            'csv': {d: save_csv for d in self.datasets},
            'excel': {d: save_xlsx for d in self.datasets}
        }
        self._file_groups = {
            'pickle': FileGroup({
                dataset: PandasPickleFile(savedir/(name+'_'+dataset+'.pkl.gz'))
                for dataset in self.datasets
            }, name=f'Pickle files for {self}'),
            'csv': FileGroup({
                dataset: PandasCSVFile(savedir/(name+'_'+dataset+'.csv'))
                for dataset in self.datasets
            }, name=f'CSV files for {self}'),
            'excel': FileGroup({dataset:
                PandasExcelFile(savedir/(name+'.xlsx'), sheet_name=dataset)
                for dataset in self.datasets
            }, name=f'Excel files for {self}')
        }
        self._file_types = ['pickle', 'csv', 'excel']

        self.__cache = {}


    @property
    def savedir(self) -> Path:
        """Default directory where data files are saved"""
        return self._savedir

    @property
    def name(self) -> str:
        """Default filename prefix for saved data files"""
        return self._name

    @property
    def description(self) -> str:
        """Description of the data"""
        return self._description


    def get(self, dataset, cached=True, saved=True):
        """
        Get dataset from cache.
        
        If `cached=False` or dataset is not in cache,
        read the dataset from pickle file and save to cache.
        If `saved=False` or dataset is not saved to pickle file,
        call :py:meth:`process(dataset)<process>` and write
        the dataset to file if auto-save is enabled.
        
        Parameters
        ----------
        dataset : str
            Name of dataset
        cached : bool, default True
            Attempt to get dataset from cache
        saved : bool, default True
            Attempt to read dataset from pickle file
        
        Returns
        -------
        pandas.DataFrame or pandas.Series
        
        Raises
        ------
        DatasetNotFoundError
            If `dataset` is not in :py:attr:`datasets`
        """
        if dataset not in self.datasets:
            raise DatasetNotFoundError(dataset, self)

        if cached:
            try:
                return self.__cache[dataset]
            except KeyError:
                pass

        if saved:
            try:
                data = self._file_groups['pickle'][dataset].read()
            except FileNotFoundError:
                pass
            else:
                self.__cache[dataset] = data
                return data

        # Get data
        data = self.process(dataset)

        # Save data to cache
        self.__cache[dataset] = data

        # Write data to file if auto-save is enabled
        for file_type in self._file_types:
            if self._auto_save[file_type][dataset]:
                file = self._file_groups[file_type][dataset]
                file.write(data)

        return data


    def process(self, dataset):
        """Produce dataset.
        Internally, this calls :py:meth:`_process_\<dataset\>`.
        
        Parameters
        ----------
        dataset : str
            Name of dataset
        
        Returns
        -------
        pandas.DataFrame or pandas.Series
        
        Raises
        ------
        DatasetNotFoundError
            If `dataset` is not in :py:attr:`datasets`
        """
        if dataset not in self.datasets:
            raise DatasetNotFoundError(dataset, self)

        return getattr(self, '_process_'+dataset)()


    def items(self):
        """Generate (key, value) pairs for the datasets.
        
        Yields
        ------
        key : str
            Name of dataset
        value : pandas.DataFrame or pandas.Series
            Dataset
        """
        for dataset in self.datasets:
            yield dataset, self.get(dataset)


    def _to_file(self, file_type, dataset=None, path=None, **kwargs):
        """Write dataset to {file_type} file.
        
        Parameters
        ----------
        dataset : str, default None
            Name of dataset. If `None`, write all datasets to file.
        path : str or pathlike, default None
            Destination path of file. If `None`, write dataset to
            its default path in :py:attr:`savedir`.
        
        Other parameters
        ----------------
        **kwargs
            Keyword arguments passed to pandas method
            :py:meth:`to_{file_type}<pandas.DataFrame.to_{file_type}>`
        """
        if isinstance(dataset,str):
            return self._dataset_to_file(file_type, dataset, path, **kwargs)
        else:
            datasets = self.datasets if dataset is None else dataset
            if len(datasets) > 1 and path is not None and file_type != 'excel':
                logger.warning(f'Datasets {datasets} will overwrite each other'
                                                f' in the same file: {path}')
            return FileGroup({
                dataset: self._dataset_to_file(
                    file_type, dataset, path, **kwargs
                ) for dataset in datasets
            })


    def _dataset_to_file(self, file_type, dataset, path=None, **kwargs):
        if file_type not in self._file_types:
            raise NotImplementedError(f'Unknown file type "{file_type}"')

        if path is None:
            file = self._file_groups[file_type][dataset]
        else:
            if file_type == 'pickle':
                file = PandasPickleFile(path=path)
            elif file_type == 'csv':
                file = PandasCSVFile(path=path)
            elif file_type == 'excel':
                file = PandasExcelFile(path=path, sheet_name=dataset)

        file.write(self.get(dataset), **kwargs)
        return file


    def to_csv(self, dataset=None, path=None, **kwargs):
        return self._to_file('csv', dataset, path, **kwargs)

    def to_excel(self, dataset=None, path=None, **kwargs):
        return self._to_file('excel', dataset, path, **kwargs)

    def to_pickle(self, dataset=None, path=None, **kwargs):
        return self._to_file('pickle', dataset, path, **kwargs)

    to_csv.__doc__ = _to_file.__doc__.format(file_type='csv')
    to_excel.__doc__ = _to_file.__doc__.format(file_type='excel')
    to_pickle.__doc__ = _to_file.__doc__.format(file_type='pickle')


    def purge_cache(self):
        """Empty data cache."""
        self.__cache = {}


    def purge_savedir(self, pattern=None, ask=True):
        """Remove files belonging to this instance of :py:class:`Data`.
        
        Parameters
        ----------
        pattern : str, default None
            Pattern of filenames to remove. If `None`, remove all
            files belonging to this instance of :py:class:`Data`.
        ask : bool, default True
            Ask for confirmation before removing.
        """
        if pattern is None:
            for file_group in self._file_groups.values():
                file_group.remove(ask=ask, missing_okay=True)
        else:
            for file_group in self._file_groups.values():
                for file in file_group:
                    if fnmatch(file.filename, pattern):
                        file.remove(ask=ask, missing_okay=True)


    def __getitem__(self, item):
        if isinstance(item,str):
            return self.get(item)
        try:
            iter(item)
        except TypeError:
            raise TypeError('Key must be str or an iterable of str')
        else:
            return [self[i] for i in item]

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name not in self.datasets:
                raise
        return self.get(name)

    def __dir__(self):
        return super().__dir__() + self.datasets

    def __repr__(self):
        info = f'savedir={self.savedir}, name="{self.name}", description='
        info += 'None' if self.description is None else f'"{self.description}"'
        return self.__class__.__name__ + '(' + info + ')'

    def __str__(self):
        return self.__class__.__name__ #+ f'("{self.name}")'

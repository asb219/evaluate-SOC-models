import pandas as pd

from data_manager import \
    Data, ZipArchive, FileFromURL, PandasCSVFile, FileFromArchive

from evaluate_SOC_models.path import DOWNLOADPATH, DATAPATH


__all__ = [
    'Graven2017Archive',
    'Graven2017CompiledRecordsFile',
    'Graven2017CompiledRecordsData'
]


class Graven2017Archive(FileFromURL, ZipArchive):
    """
    Data supplement of "Compiled records of carbon isotopes in
    atmospheric CO2 for historical simulations in CMIP6"
    Graven et al. (2017), Geosci. Model Dev., 10, 4405–4417
    https://gmd.copernicus.org/articles/10/4405/2017/
    """

    def __init__(self):
        DL_link = (
            "https://gmd.copernicus.org/articles/10/4405/2017/"
            "gmd-10-4405-2017-supplement.zip"
        )
        filename = 'graven_compiled_gmd-2017.zip'
        savedir = DOWNLOADPATH/'Graven2017'
        super().__init__(savedir / filename, DL_link)



class Graven2017CompiledRecordsFile(PandasCSVFile, FileFromArchive):
    """
    Compiled records of atmospheric 14CO2 and 13CO2 by Graven et al. (2017)
    """

    def __init__(self):
        archive = Graven2017Archive()
        archived_filename = "TableS1.csv"
        filename = 'graven_Delta14CO2_delta13CO2.csv'
        savedir = DOWNLOADPATH/'Graven2017'
        super().__init__(savedir / filename, archive, archived_filename)

    def _read(self, **kwargs):
        kw = dict(header=3, skiprows=[4], index_col=0)
        kw.update(kwargs)
        df = super()._read(**kw)
        years = df.index.astype('int').astype('str')
        df.index = pd.to_datetime(years, format='%Y')
        df.index.name = 'time'
        return df



class Graven2017CompiledRecordsData(Data):
    " Graven et al. (2017), https://gmd.copernicus.org/articles/10/4405/2017/ "


    datasets = ['delta13C', 'Delta14C', 'F14C']


    def __init__(self, *, save_pkl=True, save_csv=True, save_xlsx=False):

        savedir = DATAPATH / 'Graven2017'
        name = 'atmospheric_co2'
        description = ('Compiled records of atmospheric 14CO2 & 13CO2'
            ', by Graven et al. (2017)')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.sourcefile = Graven2017CompiledRecordsFile()


    def _process_F14C(self):
        return self['Delta14C'] / 1000 + 1

    def _process_Delta14C(self):
        zones = ['NH','Tropics','SH']
        usecols = ['Date']+[z+' Delta14co2' for z in zones]
        rename = {z+' Delta14co2':z for z in zones}
        Delta14C = self.sourcefile.read(usecols=usecols).rename(columns=rename)
        return Delta14C

    def _process_delta13C(self):
        usecols = ['Date','Global delta13co2']
        delta13C = self.sourcefile.read(usecols=usecols).squeeze('columns')
        return delta13C

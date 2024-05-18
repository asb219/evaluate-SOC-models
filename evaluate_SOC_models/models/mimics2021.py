import subprocess
import json
import urllib
from loguru import logger

from data_manager import (
    DataFile,
    FileFromDownload,
    FileFromURL,
    SevenZipArchive,
    FileGroupFromArchive,
    FileFromArchive,
    FileFrom,
    PandasCSVFile
)
from evaluate_SOC_models.path import MIMICS2021REPOSITORYPATH



class JSONFile(DataFile):

    def _read(self, **kwargs):
        with self.path.open('r') as f:
            return json.load(f)

    def _write(self, data, **kwargs):
        with self.path.open('w') as f:
            return json.dump(data, f)


class CSIRORequestJSONFile(JSONFile, FileFromDownload):
    """
    Based on tutorial:
    https://confluence.csiro.au/display/dap/DAP+Web+Services+-+Python+Examples
    """

    def __init__(self, path, url, **kwargs):
        super().__init__(path, **kwargs)
        self.url = url

    def _download(self):
        req = urllib.request.Request(self.url)
        req.add_header('Accept', 'application/json')
        response = urllib.request.urlopen(req)
        data = json.load(response)
        self.write(data)
        return str(self.path)


class MIMICS2021CodeArchive(FileFromURL, SevenZipArchive):
    """
    Wang Y-P (2020),
    "Vertically resolved soil carbon model (model codes and site data)"
    https://data.csiro.au/collection/csiro:47942v1
    
    Associated to research article:
    Wang Y-P, et al. (2021)
    "Microbial Activity and Root Carbon Inputs Are More Important than
    Soil Carbon Diffusion in Simulating Soil Carbon Profiles", JGR-BGS.
    DOI 10.1029/2020JG006205
    """

    doi = '10.25919/843a-w584'

    def __init__(self):
        self.collection_json = CSIRORequestJSONFile(
            path = MIMICS2021REPOSITORYPATH / 'CSIRO_collection.json',
            url = 'https://data.csiro.au/dap/ws/v2/collections/' + self.doi
        )
        self.filelist_json = CSIRORequestJSONFile(
            path = MIMICS2021REPOSITORYPATH / 'CSIRO_filelist.json',
            url = self.collection_json.read()['data']
        )
        filelist = self.filelist_json.read()
        info, = [f for f in filelist['file'] if f['filename'] == 'model.7z']
        super().__init__(
            path = MIMICS2021REPOSITORYPATH / 'model.7z',
            url = info['link']['href']
        )


class MIMICS2021CodeFiles(FileGroupFromArchive):

    def __init__(self):
        archive = MIMICS2021CodeArchive()
        filename_list = archive.get_filename_list()
        files = {
            filename: FileFromArchive(
                path = MIMICS2021REPOSITORYPATH / 'model' / filename,
                archive = archive,
                archived_filename = filename
            ) for filename in filename_list if filename != 'test01'
        } # exclude file 'test01' because it's a foreign executable
        super().__init__(files, name='MIMICS 2021 code files')
        self.archive = archive


class MIMICS2021ModifiedSourceCodeFile(FileFrom):

    def __init__(self):
        filename = 'vsoilmic05f_ms25_modified.f90'
        super().__init__(MIMICS2021REPOSITORYPATH / 'model' / filename)

    def fetch(self):
        self.modify_source_code()

    def modify_source_code(self):
        """
        Modify source code so that MIMICS2021 outputs 12C and 14C stocks
        for each year.
        """
        original_file = MIMICS2021CodeFiles()['vsoilmic05f_ms25.f90'].claim()
        code = original_file.path.read_text()
        modified_code = code.replace(
            "      open(91,file='modobs.txt')\n"
            ,
            "      open(91,file='modobs.txt')\n"
            "      open(66614, file='out14.txt') !!! added by asb219\n"
            "      open(66612, file='out12.txt') !!! added by asb219\n"
            ,
            1 # only replace the first instance
        ).replace(
            "      close(91)\n"
            ,
            "      close(91)\n"
            "      close(66614) !!! added by asb219\n"
            "      close(66612) !!! added by asb219\n"
            ,
            1 # only replace the first instance
        ).replace(
            "        enddo ! year\n"
            ,
            "\n"
            "          !!! added by asb219 START\n"
            "          if (year > 0) then\n"
            "            if (isoc14 == 2) then\n"
            "              write(66614,*) siteid(np), year, cpool(np,1,:)\n"
            "            elseif (isoc14 == 1) then\n"
            "              write(66612,*) siteid(np), year, cpool(np,1,:)\n"
            "            endif\n"
            "          endif\n"
            "          !!! added by asb219 END\n"
            "\n"
            "        enddo ! year\n"
            ,
            1 # only replace the first instance
        )
        self.path.write_text(modified_code)


class MIMICS2021ExecutableFile(FileFrom):

    def __init__(self):
        super().__init__(MIMICS2021REPOSITORYPATH / 'model' / 'test_modified')

    def fetch(self):
        self.compile()

    def compile(self):
        for_file = MIMICS2021CodeFiles().claim()['test01.for']
        f90_file = MIMICS2021ModifiedSourceCodeFile().claim()
        command = f'gfortran -o {self} {for_file} {f90_file}'
        return subprocess.check_call(command, shell=True)

    def execute(self):
        self.claim()
        logger.debug(f'Executing MIMICS2021: {self}')
        subprocess.check_call(f'"{self}"', cwd=self.savedir, shell=True)
        logger.debug('MIMICS2021 execution finished, exit code 0.')


class MIMICS2021OutputFile(PandasCSVFile, FileFrom):

    pools = ['LIT_m', 'LIT_s', 'MIC_r', 'MIC_K', 'SOM_p', 'SOM_c', 'SOM_a']

    def __init__(self, isotope, readonly=True, **kwargs):
        if isotope == '12C':
            filename = 'out12.txt'
        elif isotope == '14C':
            filename = 'out14.txt'
        else:
            raise ValueError
        path = MIMICS2021REPOSITORYPATH / 'model' / filename
        super().__init__(path, readonly=readonly, **kwargs)

    def fetch(self):
        MIMICS2021ExecutableFile().execute()

    def _read(self, **kwargs):
        kwargs.setdefault('sep', '\s+')
        kwargs.setdefault('names', ['siteid','year'] + self.pools)
        df = super()._read(**kwargs)
        if 'year' in df.columns:
            df['year'] += 950
        return df

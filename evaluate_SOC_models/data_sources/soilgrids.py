import numpy as np
import pandas as pd
from soilgrids import SoilGrids

from evaluate_SOC_models.data_manager.data import Data
from evaluate_SOC_models.data_manager.file import XarrayGeoTIFFile, FileFromDownload

from evaluate_SOC_models.path import DOWNLOADPATH, DATAPATH


__all__ = [
    'SoilGridsFile',
    'SoilGridsPointFile',
    'SoilGridsPointData'
]


class SoilGridsFile(XarrayGeoTIFFile, FileFromDownload):
    """
    SoilGrids, https://www.isric.org/explore/soilgrids
    Interactive map: https://soilgrids.org
    """

    @staticmethod
    def print_map_services():
        SoilGrids().map_services

    def __init__(self, coverage_id, west, south, east, north, *,
            width=50, height=50, crs='urn:ogc:def:crs:EPSG::4326'):

        identifier = 'lat{}-{}_lon{}-{}_w{}_h{}'.format(
            south, north, west, east, width, height
        )
        savedir = DOWNLOADPATH / 'SoilGrids' / identifier
        filename = coverage_id + '.tif'

        super().__init__(savedir / filename)

        service_id, depth_id, mean_std_Q = coverage_id.split('_')

        self.coverage_id = coverage_id # e.g. 'clay_0-5cm_mean'
        self.service_id = service_id # e.g. 'clay'
        self.depth_id = depth_id # e.g. '0-5cm'
        self.west = west
        self.south = south
        self.east = east
        self.north = north
        self.width = width
        self.height = height
        self.crs = crs
        self.soilgrids = SoilGrids()


    def _read(self, **kwargs):
        try:
            return self.__cached_data
        except AttributeError:
            return super()._read(**kwargs)

    def _download(self):
        data = self.soilgrids.get_coverage_data(
            service_id=self.service_id, coverage_id=self.coverage_id,
            west=self.west, south=self.south,
            east=self.east, north=self.north,
            width=self.width, height=self.height,
            crs=self.crs, output=str(self.path)
        )
        self.__cached_data = data # xarray DataArray
        return self.soilgrids.tif_file # tif file path

    @property
    def _tmp_filename(self):
        return self.filename



class SoilGridsPointFile(SoilGridsFile):

    def __init__(self, coverage_id, lat, lon, *, dx=0.02, dy=0.02,
            width=10, height=10, crs='urn:ogc:def:crs:EPSG::4326'):

        super().__init__(coverage_id,
            west=lon-dx/2, south=lat-dy/2, east=lon+dx/2, north=lat+dy/2,
            width=width, height=height, crs=crs
        )

        identifier = f'lat{lat}_lon{lon}_dx{dx}_dy{dy}_w{width}_h{height}'
        self.savedir = DOWNLOADPATH / 'SoilGridsPoint' / identifier

        self.lat = lat
        self.lon = lon



class SoilGridsPointData(Data):
    " SoilGrids, https://www.isric.org/explore/soilgrids "

    datasets = ['layer','integrated'] + [d+v for d in ('layer_','integrated_')
        for v in ['bd','pH','sand','silt','clay']]

    variables = ['bd','pH','sand','silt','clay']
    service_id_dict = {
        'bd':'bdod', 'pH':'phh2o', 'sand':'sand', 'silt':'silt', 'clay':'clay'
    }
    #depth_layer_bounds = [0,5,15,30,60,100,200]


    def __init__(self, lat, lon, top, bot, *, # top,bot (cm) of soil layer
            dx=0.01, dy=0.01, width=10, height=10, # SoilGrids download kwargs
            save_pkl=False, save_csv=False, save_xlsx=False):

        if lon>=180:
            lon -= 360 # negative degrees west instead of degress east

        lat = np.round(float(lat),3)
        lon = np.round(float(lon),3)
        top = int(np.round(top,0))
        bot = int(np.round(bot,0))

        assert 0 <= top < bot <= 200

        identifier = f'lat{lat:+07.3f}_lon{lon:+08.3f}_top{top}_bot{bot}'
        savedir = DATAPATH / 'SoilGrids' / identifier
        name = 'sg'
        description = f'SoilGrids data for {top}-{bot}cm depth at {lat}N, {lon}E.'

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.sourcefiles = dict()

        for variable in self.variables:
            service_id = self.service_id_dict[variable]
            files = self._get_TIF_files(
                service_id, lat, lon, top, bot,
                dx=dx, dy=dy, width=width, height=height
            )
            self.sourcefiles[variable] = files

        self.lat = lat
        self.lon = lon
        self.top = top
        self.bot = bot


    def _process_layer(self):
        return pd.DataFrame({v: self['layer_'+v] for v in self.variables})

    def _process_layer_bd(self):
        return self._get_layer_data('bd') / 100 # cg/cm3 to g/cm3 = tonnes/m3

    def _process_layer_pH(self):
        return self._get_layer_data('pH') / 10 # change scale of 0-140 to 0-14

    def _process_layer_sand(self):
        return self._get_layer_data('sand') / 10 # g/kg to percent

    def _process_layer_silt(self):
        return self._get_layer_data('silt') / 10 # g/kg to percent

    def _process_layer_clay(self):
        return self._get_layer_data('clay') / 10 # g/kg to percent

    def _get_layer_data(self, variable):
        return pd.Series({ # average data over (horizontal) space
            f.depth_id: (data:=f.read()).where(data>0, np.nan).mean().item()
            for f in self.sourcefiles[variable] # for each depth layer
        })


    def _process_integrated(self):
        return pd.concat([self['integrated_'+v] for v in self.variables])

    def _process_integrated_bd(self):
        return self._get_depth_integrated_data('bd')

    def _process_integrated_pH(self):
        return self._get_depth_integrated_data('pH')

    def _process_integrated_sand(self):
        return self._get_depth_integrated_data('sand')

    def _process_integrated_silt(self):
        return self._get_depth_integrated_data('silt')

    def _process_integrated_clay(self):
        return self._get_depth_integrated_data('clay')

    def _get_depth_integrated_data(self, variable):
        try:
            layer_thicknesses = self.__layer_thicknesses
        except AttributeError:
            depth_ids = self._get_depth_ids(self.top, self.bot)
            depth_intervals = self._get_depth_intervals(self.top, self.bot)
            depth_intervals[0][0] = self.top
            depth_intervals[-1][1] = self.bot
            layer_thicknesses = pd.Series({
                depth_id: d2-d1 for depth_id, (d1,d2)
                in zip(depth_ids, depth_intervals)
            })
            self.__layer_thicknesses = layer_thicknesses
        if variable == 'bd':
            integrator = layer_thicknesses
        else:
            integrator = self['layer_bd'] * layer_thicknesses # = bulk mass
        value = self['layer_'+variable].dot(integrator) / integrator.sum()
        return pd.Series([value], index=[variable])


    def _get_TIF_files(self, service_id, lat, lon, top, bot, **kwargs):
        coverage_ids = self._get_coverage_ids(service_id, top, bot)
        files = [
            SoilGridsPointFile(coverage_id, lat, lon, **kwargs)
            for coverage_id in coverage_ids
        ]
        return files

    def _get_coverage_ids(self, service_id, top, bot):
        depth_ids = self._get_depth_ids(top, bot)
        coverage_ids = [service_id+'_'+d+'_mean' for d in depth_ids]
        return coverage_ids

    def _get_depth_ids(self, top, bot):
        depth_intervals = self._get_depth_intervals(top, bot)
        depth_ids = [f'{d1}-{d2}cm' for d1,d2 in depth_intervals]
        return depth_ids

    def _get_depth_intervals(self, top, bot):
        depth_intervals = []
        depth_bounds = [0,5,15,30,60,100,200]
        for d1,d2 in zip(depth_bounds[:-1], depth_bounds[1:]):
            if top < d2 and bot > d1:
                depth_intervals.append([d1,d2])
        return depth_intervals

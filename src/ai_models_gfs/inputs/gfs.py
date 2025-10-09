# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import os
import eccodes
import numpy as np
import earthkit.data as ekd
import urllib.request
import tempfile
from datetime import datetime,timedelta
from scipy.interpolate import RegularGridInterpolator
from .base import RequestBasedInput

LOG = logging.getLogger(__name__)

class GfsInput(RequestBasedInput):
    WHERE = "GFS"
    def pl_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))
        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_pres.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_pres.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_pressure_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted pressure data
        formatted_pressure_file = (
            f"/tmp/ai-models-gfs/gfspresformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")

        formatted_pressure_output = ekd.new_grib_output(
            formatted_pressure_file, edition=1
        )
        # Construct the URL to fetch GFS pressure data
        gfs_pressure_url = (
#            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/"
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{str(kwargs['date'])}/"           
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS pressure data from the URL
        gfs_pressure_data = ekd.from_source("url", gfs_pressure_url)

        # Iterate over the sample pressure GRIB messages
        for grib_message in sample_pressure_grib:
            parameter_name = grib_message['shortName']
            pressure_level = grib_message['level']
            template = grib_message
            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "z":
                # Select geopotential height data and convert to meters
                geopotential_height_data = gfs_pressure_data.sel(
                    param="gh", level=pressure_level
                )
                data_array = geopotential_height_data[0].to_numpy() * 9.80665
            else:
                # Select other parameters' data
                parameter_data = gfs_pressure_data.sel(
                    param=parameter_name, level=pressure_level
                )
                data_array = parameter_data[0].to_numpy()
            if interp:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.25,-0.25),
                                        np.arange(0,360,0.25),
                                        np.arange(90,-91,-1),
                                        np.arange(0,360,1)
                                        )
            if interp:
                template = set_eccodes(template)
            # Write the data to the formatted GRIB file using the template
            formatted_pressure_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_pressure_grib = ekd.from_source("file", formatted_pressure_file)
        return formatted_pressure_grib

    def sfc_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))

        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_sfc.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_sfc.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_surface_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted surface data
        formatted_surface_file = (
            f"/tmp/ai-models-gfs/gfssfcformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )

        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")

        formatted_surface_output = ekd.new_grib_output(
            formatted_surface_file, edition=1
        )
        # Construct the URL to fetch GFS surface data
        gfs_surface_url = (
#            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/"
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{str(kwargs['date'])}/"                        
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS surface data from the URL
        gfs_surface_data = ekd.from_source("url", gfs_surface_url)

        # Iterate over the sample surface GRIB messages
        for grib_message in sample_surface_grib:
            parameter_name = grib_message['shortName']
            surface_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )
            if parameter_name == "tp":
                # For total precipitation, create an array of zeros
                data_array = np.zeros((721, 1440))
            elif parameter_name in ["z", "lsm"]:
                # For geopotential height and land-sea mask, use the data directly
                data_array = grib_message.to_numpy()
            elif parameter_name == "msl":
                # Select mean sea level pressure data
                mean_sea_level_pressure_data = gfs_surface_data.sel(param="prmsl")
                data_array = mean_sea_level_pressure_data[0].to_numpy()
            elif parameter_name == "tcwv":
                # Select total column water vapor data
                total_column_water_vapor_data = gfs_surface_data.sel(param="pwat")
                data_array = total_column_water_vapor_data[0].to_numpy()
            else:
                # Select other parameters' data
                parameter_data = gfs_surface_data.sel(param=parameter_name)
                data_array = parameter_data[0].to_numpy()
            if interp:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.25,-0.25),
                                        np.arange(0,360,0.25),
                                        np.arange(90,-91,-1),
                                        np.arange(0,360,1)
                                        )
            if interp:
                template = set_eccodes(template)            
            # Write the data to the formatted GRIB file using the template
            formatted_surface_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_surface_grib = ekd.from_source("file", formatted_surface_file)
        return formatted_surface_grib

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

class GefsInput(RequestBasedInput):
    WHERE = "GEFS"

    def __init__(self, owner, **kwargs):
        super().__init__(owner, **kwargs)
        self.kwargs = kwargs

    def pl_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))
        member = str(self.kwargs['member'][0]).zfill(2)
        if member=='00':
            member = 'c00'
        else:
            member = f'p{member}'

        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_pres.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_pres.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_pressure_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted pressure data
        formatted_pressure_file = (
            f"/tmp/ai-models-gfs/gefspresformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}_{member}.grib"
        )

        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")

        formatted_pressure_output = ekd.new_grib_output(
            formatted_pressure_file, edition=1
        )
        # Construct the URL to fetch GFS pressure data

        gefs_pressure_url_a = (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/pgrb2ap5/"
            f"ge{member}.t{str(kwargs['time']).zfill(2)}z.pgrb2a.0p50.f000"
        )

        gefs_pressure_url_b = (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/pgrb2bp5/"
            f"ge{member}.t{str(kwargs['time']).zfill(2)}z.pgrb2b.0p50.f000"
        )
        print(gefs_pressure_url_a)
        # Load the GFS pressure data from the URL
        gefs_pressure_data_a = ekd.from_source("url", gefs_pressure_url_a)
        gefs_pressure_data_b = ekd.from_source("url", gefs_pressure_url_b)
        gefs_pressure_data = gefs_pressure_data_a + gefs_pressure_data_b

        # Iterate over the sample pressure GRIB messages
        for grib_message in sample_pressure_grib:
            parameter_name = grib_message['shortName']
            pressure_level = grib_message['level']
            template = grib_message
            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "z":
                # Select geopotential height data and convert to meters
                geopotential_height_data = gefs_pressure_data.sel(
                    param="gh", level=pressure_level
                )
                data_array = geopotential_height_data[0].to_numpy() * 9.80665
            else:
                # Select other parameters' data
                parameter_data = gefs_pressure_data.sel(
                    param=parameter_name, level=pressure_level
                )
                data_array = parameter_data[0].to_numpy()

            if interp:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.50,-0.50),
                                        np.arange(0,360,0.50),
                                        np.arange(90,-91,-1),
                                        np.arange(0,360,1)
                                        )
            else:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.50,-0.50),
                                        np.arange(0,360,0.50),
                                        np.arange(90,-90.25,-0.25),
                                        np.arange(0,360,0.25)
                                        )
            # Write the data to the formatted GRIB file using the template
            if interp:
                template = set_eccodes(template)
            formatted_pressure_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_pressure_grib = ekd.from_source("file", formatted_pressure_file)
        return formatted_pressure_grib

    def sfc_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))
        member = str(self.kwargs['member'][0]).zfill(2)
        if member=='00':
            member = 'c00'
        else:
            member = f'p{member}'

        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_sfc.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_sfc.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_surface_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted surface data
        formatted_surface_file = (
            f"/tmp/ai-models-gfs/gfssfcformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}_{member}.grib"
        )

        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")

        formatted_surface_output = ekd.new_grib_output(
            formatted_surface_file, edition=1
        )

        # Construct the URL to fetch GFS surface data
        gefs_surface_url_a = (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/pgrb2ap5/"
            f"ge{member}.t{str(kwargs['time']).zfill(2)}z.pgrb2a.0p50.f000"
        )

        # Construct the URL to fetch GFS surface data
        gefs_surface_url_b = (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/pgrb2bp5/"
            f"ge{member}.t{str(kwargs['time']).zfill(2)}z.pgrb2b.0p50.f000"
        )

        # Load the GFS surface data from the URL
        gefs_surface_data_a = ekd.from_source("url", gefs_surface_url_a)
        gefs_surface_data_b = ekd.from_source("url", gefs_surface_url_b)
        gefs_surface_data = gefs_surface_data_a + gefs_surface_data_b
        # Iterate over the sample surface GRIB messages

        for grib_message in sample_surface_grib:
            parameter_name = grib_message['shortName']
            surface_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "tp":
                # For total precipitation, create an array of zeros
                data_array = np.zeros((721, 1440))
                if interp:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.25,-0.25),
                                            np.arange(0,360,0.25),
                                            np.arange(90,-91,-1),
                                            np.arange(0,360,1)
                                            )
            elif parameter_name in ["z", "lsm"]:
                # For geopotential height and land-sea mask, use the data directly
                data_array = grib_message.to_numpy()
                if interp:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.25,-0.25),
                                            np.arange(0,360,0.25),
                                            np.arange(90,-91,-1),
                                            np.arange(0,360,1)
                                            )
            elif parameter_name == "msl":
                # Select mean sea level pressure data
                mean_sea_level_pressure_data = gefs_surface_data.sel(param="prmsl")
                data_array = mean_sea_level_pressure_data[0].to_numpy()
                if interp:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-91,-1),
                                            np.arange(0,360,1)
                                            )
                else:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-90.25,-0.25),
                                            np.arange(0,360,0.25)
                                            )
            elif parameter_name == "tcwv":
                # Select total column water vapor data
                total_column_water_vapor_data = gefs_surface_data.sel(param="pwat")
                data_array = total_column_water_vapor_data[0].to_numpy()
                if interp:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-91,-1),
                                            np.arange(0,360,1)
                                            )
                else:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-90.25,-0.25),
                                            np.arange(0,360,0.25)
                                            )
            else:
                # Select other parameters' data
                parameter_data = gefs_surface_data.sel(param=parameter_name)
                data_array = parameter_data[0].to_numpy()
                if interp:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-91,-1),
                                            np.arange(0,360,1)
                                            )
                else:
                    data_array = interpolate(
                                            data_array,
                                            np.arange(90,-90.50,-0.50),
                                            np.arange(0,360,0.50),
                                            np.arange(90,-90.25,-0.25),
                                            np.arange(0,360,0.25)
                                            )



            # Write the data to the formatted GRIB file using the template
            if interp:
                template = set_eccodes(template)
            formatted_surface_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_surface_grib = ekd.from_source("file", formatted_surface_file)
        return formatted_surface_grib

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

class GdasInput(RequestBasedInput):
    WHERE = "GDAS"

    def pl_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))
        # Load the sample pressure GRIB file
        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_pres.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_pres.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_pressure_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted pressure data
        formatted_pressure_file = (
            f"/tmp/ai-models-gfs/gdaspresformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")
        formatted_pressure_output = ekd.new_grib_output(
            formatted_pressure_file, edition=1
        )
        # Construct the URL to fetch GFS pressure data
        gdas_pressure_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gdas.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        print(gdas_pressure_url)
        # Load the GFS pressure data from the URL
        gdas_pressure_data = ekd.from_source("url", gdas_pressure_url)

        # Iterate over the sample pressure GRIB messages
        for grib_message in sample_pressure_grib:
            parameter_name = grib_message['shortName']
            pressure_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "z":
                # Select geopotential height data and convert to meters
                geopotential_height_data = gdas_pressure_data.sel(
                    param="gh", level=pressure_level
                )
                data_array = geopotential_height_data[0].to_numpy() * 9.80665
            else:
                # Select other parameters' data
                parameter_data = gdas_pressure_data.sel(
                    param=parameter_name, level=pressure_level
                )
                data_array = parameter_data[0].to_numpy()

            if interp:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.25,-0.25),
                                        np.arange(0,360,0.25),
                                        np.arange(90,-91,-1),
                                        np.arange(0,360,1)
                                        )
            # Write the data to the formatted GRIB file using the template
            if interp:
                template = set_eccodes(template)
            formatted_pressure_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_pressure_grib = ekd.from_source("file", formatted_pressure_file)
        return formatted_pressure_grib

    def sfc_load_source(self, **kwargs):
        interp = bool(int(kwargs['grid'][0]))
        # Use a named temp file to ensure it persists
        temp_path = os.path.join(tempfile.gettempdir(), "sample_sfc.grib")
        sample_url = "https://noaa-oar-mlwp-data.s3.amazonaws.com/colab_resources/sample_sfc.grib"

        # Check if the file already exists before downloading
        if not os.path.exists(temp_path):
            print(f"Downloading {sample_url} to {temp_path}...")
            urllib.request.urlretrieve(sample_url, temp_path)
        else:
            print(f"File already exists: {temp_path}, skipping download.")

        # Download the file to the temp directory
        urllib.request.urlretrieve(sample_url, temp_path)

        # Load the GRIB file from the temp directory
        sample_surface_grib = ekd.from_source("file", temp_path)

        # Create a new GRIB output file for the formatted surface data
        formatted_surface_file = (
            f"/tmp/ai-models-gfs/gdassfcformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")
        formatted_surface_output = ekd.new_grib_output(
            formatted_surface_file, edition=1
        )
        # Construct the URL to fetch GFS surface data
        gdas_surface_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gdas.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS surface data from the URL
        gdas_surface_data = ekd.from_source("url", gdas_surface_url)

        # Iterate over the sample surface GRIB messages
        for grib_message in sample_surface_grib:
            parameter_name = grib_message['shortName']
            surface_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle._handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle._handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "tp":
                # For total precipitation, create an array of zeros
                data_array = np.zeros((721, 1440))
            elif parameter_name in ["z", "lsm"]:
                # For geopotential height and land-sea mask, use the data directly
                data_array = grib_message.to_numpy()
            elif parameter_name == "msl":
                # Select mean sea level pressure data
                mean_sea_level_pressure_data = gdas_surface_data.sel(param="prmsl")
                data_array = mean_sea_level_pressure_data[0].to_numpy()
            elif parameter_name == "tcwv":
                # Select total column water vapor data
                total_column_water_vapor_data = gdas_surface_data.sel(param="pwat")
                data_array = total_column_water_vapor_data[0].to_numpy()
            else:
                # Select other parameters' data
                parameter_data = gdas_surface_data.sel(param=parameter_name)
                data_array = parameter_data[0].to_numpy()

            if interp:
                data_array = interpolate(
                                        data_array,
                                        np.arange(90,-90.25,-0.25),
                                        np.arange(0,360,0.25),
                                        np.arange(90,-91,-1),
                                        np.arange(0,360,1)
                                        )
            # Write the data to the formatted GRIB file using the template
            if interp:
                template = set_eccodes(template)
            formatted_surface_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_surface_grib = ekd.from_source("file", formatted_surface_file)
        return formatted_surface_grib

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

def interpolate(data,inlats,inlons,outlats,outlons):
    inlons_extended = np.concatenate(([inlons[-1] - 360], inlons, [inlons[0] + 360]))
    data_extended = np.concatenate((data[:, -1:], data, data[:, :1]), axis=1)
    interpolator = RegularGridInterpolator((inlats, inlons_extended), data_extended, bounds_error=False)
    outlon_grid,outlat_grid = np.meshgrid(outlons,outlats)
    points = np.array([outlat_grid.flatten(),outlon_grid.flatten()]).T
    data_interpolated = interpolator(points).reshape(outlat_grid.shape)
    return data_interpolated

def set_eccodes(template):
    grib_handle = template.handle._handle
    eccodes.codes_set(grib_handle, "Ni", 360)  # Longitude points
    eccodes.codes_set(grib_handle, "Nj", 181)  # Latitude points

    # Set correct grid spacing for 1-degree resolution
    eccodes.codes_set(grib_handle, "iDirectionIncrementInDegrees", 1.0)
    eccodes.codes_set(grib_handle, "jDirectionIncrementInDegrees", 1.0)

    # Define latitude/longitude bounds
    eccodes.codes_set(grib_handle, "latitudeOfFirstGridPointInDegrees", 90)
    eccodes.codes_set(grib_handle, "longitudeOfFirstGridPointInDegrees", 0)
    eccodes.codes_set(grib_handle, "latitudeOfLastGridPointInDegrees", -90)
    eccodes.codes_set(grib_handle, "longitudeOfLastGridPointInDegrees", 359)
    return template

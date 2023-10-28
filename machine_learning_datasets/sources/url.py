"""Load Dataset from URL"""
# pylint: disable=E1101,W0611,W0613,C0103
from typing import Dict, Any
import os
import re
import json
import warnings
from zipfile import ZipFile
import requests
from .source import Source

class URL(Source):
    """Load Dataset from URL Class"""

    def fetch(
            self,
            **kwargs:Any
        ) -> Dict:
        """Fetches a file from a given location and saves it locally.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the modified keyword arguments.

        Raises:
            Warning: If the zip file is not found at the specified path.
        """
        nkwargs = locals()['kwargs']
        nkwargs['path'] = False
        if 'location' in nkwargs:
            dssave_path = self.mlds.config.dssave_path
            if not os.path.exists(dssave_path):
                os.makedirs(dssave_path)
            location = nkwargs['location']
            r = requests.get(location, allow_redirects=True, timeout=60)
            if 'saveas' in nkwargs:
                filename = nkwargs['saveas']
            else:
                filename = location.rsplit('/', 1)[1]
                if len(filename) == 0:
                    filename = r.headers.get('content-disposition')
                    if filename is not None:
                        filename = re.findall('filename=(.+)', filename)
            zipfile_path = os.path.join(dssave_path, filename)
            open(zipfile_path, 'wb').write(r.content)
            print(location + ' downloaded to ' + zipfile_path)
            if r.headers.get('Content-Type') == 'application/zip' or\
                filename.rsplit('.', 1)[1] == 'zip':
                dsname = filename.rsplit('.', 1)[0]
                unzipdir_path = os.path.join(dssave_path, dsname)
                if os.path.exists(zipfile_path):
                    with ZipFile(zipfile_path, 'r') as zipObj:
                        zipObj.extractall(unzipdir_path)
                        print(zipfile_path + ' uncompressed to ' + unzipdir_path)
                        nkwargs['path'] = unzipdir_path
                else:
                    warnings.warn("Zip file not found at " + zipfile_path)
            else:
                nkwargs['path'] = os.path.dirname(zipfile_path)
                nkwargs['filenames'] = os.path.basename(zipfile_path)
        return nkwargs

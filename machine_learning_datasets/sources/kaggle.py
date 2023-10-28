"""ML Dataset from Kaggle"""
# pylint: disable=E1101,W0231,W0613,C3001
from typing import Optional, Dict, Any
import os
import json
import warnings
import sys
import re
from zipfile import ZipFile
from ..common import runcmd
from .source import Source

class Kaggle(Source):
    """ML Dataset from Kaggle Class"""

    def __init__(
            self,
            enabled:Optional[bool] = None,
            creds_json_path:Optional[str] = None,
            creds_username:Optional[str] = None,
            creds_key:Optional[str] = None
        ) -> None:

        self.enabled = False
        self.mlds = sys.modules['.'.join(__name__.split('.')[:-2]) or '__main__']

        if (enabled is None) or (enabled is True):
            if creds_json_path is not None:
                self.creds_json_path = creds_json_path
                with open(creds_json_path, encoding="utf-8") as file:
                    creds = json.load(file)
                    creds_username = creds['username']
                    creds_key = creds['key']
            if creds_username is not None:
                self.creds_username = creds_username
            else:
                self.creds_username = None
            if creds_key is not None:
                self.creds_key = creds_key
            else:
                self.creds_key = None
            if self.creds_username is not None and self.creds_key is not None:
                os.environ['KAGGLE_USERNAME'] = self.creds_username
                os.environ['KAGGLE_KEY'] = self.creds_key
            if 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
                self.enabled = True
                test_cmd = 'kaggle datasets list --sort-by votes'
                error, output, _ = runcmd(test_cmd)
                if error:
                    self.enabled = False
                    warnings.warn("Kaggle error! "+output)
            #TODO: what to do in this case?
            #else:
            #    warnings.warn("No Kaggle credentials set! Could not find kaggle.json.
            #    Make sure it's located in ~/.kaggle. Or use the environment method.")

    def fetch(
            self,
            **kwargs:Any
        ) -> Dict:
        """Fetch data from Kaggle datasets.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Dict: A dictionary containing the modified keyword arguments.

        Raises:
            Warning: If the zip file or non-zip file is not found.

        Example:
            >>> fetch(location='username/dataset', filenames=['file1.csv', 'file2.csv'])
        """
        nkwargs = locals()['kwargs']
        nkwargs['path'] = False
        if 'location' in nkwargs:
            dssave_path = self.mlds.config.dssave_path
            if not os.path.exists(dssave_path):
                os.makedirs(dssave_path)
            location = nkwargs['location']
            single_file = False
            single_filename = ""
            if isinstance(nkwargs['filenames'], str):
                single_file = True
                single_filename = nkwargs['filenames']
            elif isinstance(nkwargs['filenames'], list) and len(nkwargs['filenames']) == 1:
                single_file = True
                single_filename = nkwargs['filenames'][0]
            if single_file:
                cmd = f'kaggle datasets download {location} -f "{single_filename}" '\
                      f'-p "{dssave_path}"'
            else:
                cmd = f'kaggle datasets download {location} -p "{dssave_path}"'
            error, output, _ = runcmd(cmd, True)
            if not error:
                search_str = "Downloading (.*) to |^(.*): Skipping, found more recently "\
                             "modified local copy"
                zipname_match = re.search(search_str, output, re.MULTILINE)
                if zipname_match is not None:
                    conv = lambda i: i or ''
                    zipname = ''.join([conv(i) for i in zipname_match.groups()])
                    zipfile_path = os.path.join(dssave_path, zipname)
                    if re.match('.*\.zip$', zipname) is not None:
                        dsname = location.split('/')[1]
                        unzipdir_path = os.path.join(dssave_path, dsname)
                        if os.path.exists(zipfile_path):
                            with ZipFile(zipfile_path, 'r') as zip_obj:
                                zip_obj.extractall(unzipdir_path)
                                print(zipfile_path + ' uncompressed to ' + unzipdir_path)
                                nkwargs['path'] = unzipdir_path
                        else:
                            warnings.warn("Zip file not found at "+zipfile_path)
                    elif single_file and os.path.exists(zipfile_path):
                        if not os.path.isdir(zipfile_path):
                            nkwargs['path'] = os.path.dirname(zipfile_path)
                            nkwargs['filenames'] = zipfile_path.rsplit('/', 1)[1]
                        else:
                            nkwargs['path'] = zipfile_path
                    else:
                        warnings.warn("Non-zip file not found at "+zipfile_path)
                else:
                    warnings.warn("File not saved at "+dssave_path)

        return nkwargs

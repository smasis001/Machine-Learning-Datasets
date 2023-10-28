"""ML Dataset Container"""
from typing import Optional
import os
import json

class MLdatasets:
    """ML Dataset Container Class"""
    dsconfig_path:str = None
    kaggle_enabled:bool = None
    kaggle_json_path:str = None
    kaggle_username:str = None
    kaggle_key:str = None

    def __init__(
            self,
            dsconfig_path:Optional[str] = './dsconfig.json',
            kaggle_enabled:Optional[bool] = None,
            kaggle_json_path:Optional[str] = None,
            kaggle_username:Optional[str] = None,
            kaggle_key:Optional[str] = None
        ) -> None:

        self.dsconfig_path = dsconfig_path
        self.kaggle_enabled = False

        if (kaggle_enabled is None) or (kaggle_enabled is True):
            if kaggle_json_path is not None:
                self.kaggle_json_path = kaggle_json_path
                with open('./kaggle.json', encoding="utf-8") as file:
                    kaggle_creds = json.load(file)
                    kaggle_username = kaggle_creds['username']
                    kaggle_key = kaggle_creds['key']
            if kaggle_username is not None:
                self.kaggle_username = kaggle_username
            if kaggle_key is not None:
                self.kaggle_key = kaggle_key
            if self.kaggle_username is not None and self.kaggle_key is not None:
                os.environ['KAGGLE_USERNAME'] = self.kaggle_username
                os.environ['KAGGLE_KEY'] = self.kaggle_key
            if 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
                self.kaggle_enabled = True
                #test_cmd = 'kaggle datasets list --sort-by votes'
                #runcmd(test_cmd)
                #os.environ['KAGGLE_USERNAME'] = self.kaggle_username
                #os.environ['KAGGLE_KEY'] = self.kaggle_key
        return None

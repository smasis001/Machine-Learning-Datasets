"""ML Dataset Source Class"""
# pylint: disable=E1101,W0611,W0613,W0122,W0123,C0123
from typing import Any, Dict, Tuple
import sys
import warnings
import re
import os
import glob
import json
import pandas as pd
import numpy as np
import cv2
from ..preprocess import make_dummies_with_limits, make_dummies_from_dict

class Source:
    """ML Dataset Source Class"""

    def __init__(
            self,
            enabled=True
        ) -> None:
        self.enabled = enabled
        self.mlds = sys.modules['.'.join(__name__.split('.')[:-2]) or '__main__']

    def extract(
            self,
            **kwargs:Any
        ) -> Dict:
        """Extract files from a specified path.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Dict: A dictionary containing the extracted files.

        Raises:
            Warning: If 'filenames' and 'filetypes' are not of the same type in the dsconfig JSON
                     file.
            Warning: If 'filenames' and 'filetypes' in the dsconfig JSON file are not strings or
                     lists.
            Warning: If 'filenames' and 'filetypes' in the dsconfig JSON file do not have the same
                     list length.
        """
        nkwargs = locals()['kwargs']
        nkwargs['files'] = False
        if 'path' in nkwargs and 'filenames' in nkwargs and 'filetypes' in nkwargs:
            if type(nkwargs['filenames']) != type(nkwargs['filetypes']):
                warnings.warn("In dsconfig JSON file filenames and filetypes must be the same type")
            else:
                if isinstance(nkwargs['filenames'], str):
                    nkwargs['filenames'] = [nkwargs['filenames']]
                    nkwargs['filetypes'] = [nkwargs['filetypes']]
                if not isinstance(nkwargs['filenames'], list):
                    warnings.warn("In dsconfig JSON file filenames and filetypes must be "
                                  "strings or lists")
                elif len(nkwargs['filenames']) != len(nkwargs['filetypes']):
                    warnings.warn("In dsconfig JSON file filenames and filetypes must be the "
                                  "same list length")
                else:
                    if 'filesplits' in nkwargs:
                        if isinstance(nkwargs['filesplits'], str):
                            nkwargs['filesplits'] = [nkwargs['filesplits']]
                        if not isinstance(nkwargs['filenames'], list) or\
                            len(nkwargs['filenames']) != len(nkwargs['filesplits']):
                            del nkwargs['filesplits']
                    if 'filesplits' not in nkwargs:
                        nkwargs['filesplits'] = ['general'] * len(nkwargs['filenames'])
                    nkwargs['files'] = []
                    for i in range(len(nkwargs['filenames'])):
                        filename = nkwargs['filenames'][i]
                        filetype = nkwargs['filetypes'][i]
                        filesplit = nkwargs['filesplits'][i]
                        if re.search('\*', filename) is not None:
                            filepath = os.path.join(nkwargs['path'], filename)
                            files = [{'filetype':filetype, 'filesplit':filesplit,\
                                      'filename':filename,\
                                '__dirname__':os.path.basename(os.path.dirname(f)),\
                                '__filename__':os.path.basename(f),\
                                '__filepath__':f} for f in glob.glob(filepath, recursive=True)]
                            nkwargs['files'].extend(files)
                        else:
                            filepath = os.path.join(nkwargs['path'], filename)
                            if os.path.exists(filepath):
                                dirname = os.path.basename(os.path.dirname(filepath))
                                fname = os.path.basename(filepath)
                                nkwargs['files'].append({'filetype':filetype,'filesplit':filesplit,\
                                                    'filename':filename,'__dirname__':dirname,\
                                                    '__filename__':fname,'__filepath__':filepath})
                    del nkwargs['filenames']
                    del nkwargs['filetypes']
                    #del nkwargs['filesplits']
                    print(f"{len(nkwargs['files'])} dataset files found "
                          f"in {nkwargs['path']} folder")

        return nkwargs

    def parse(
            self,
            **kwargs:Any
        ) -> Dict:
        """Parses the given files based on their filetypes and options.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the parsed files and their contents.
        """
        nkwargs = locals()['kwargs']
        if 'files' in nkwargs and len(nkwargs['files']):
            for i in range(len(nkwargs['files'])):
                file = nkwargs['files'][i]
                if file['filetype'] == 'csv':
                    if 'csvopts' not in nkwargs:
                        nkwargs['csvopts'] = {}
                    if 'sep' not in nkwargs['csvopts']:
                        nkwargs['csvopts']['sep'] = ','
                    if 'removecols' in nkwargs:
                        removecols = nkwargs['removecols'].copy()
                        nkwargs['csvopts']['usecols'] = lambda x: x not in removecols
                        del nkwargs['removecols']
                    nkwargs['files'][i]['content'] = self.parse_csv(file['__filepath__'],\
                                                                    nkwargs['csvopts'])
                elif file['filetype'] == 'xls':
                    if 'xlsopts' not in nkwargs:
                        nkwargs['xlsopts'] = {}
                    if 'removecols' in nkwargs:
                        removecols = nkwargs['removecols'].copy()
                        nkwargs['xlsopts']['usecols'] = lambda x: x not in removecols
                        del nkwargs['removecols']
                    nkwargs['files'][i]['content'] = self.parse_xls(file['__filepath__'],\
                                                                    nkwargs['xlsopts'])
                elif file['filetype'] == 'img':
                    if 'imgopts' not in nkwargs:
                        nkwargs['imgopts'] = {}
                    nkwargs['files'][i]['content'] = self.parse_img(file['__filepath__'],\
                                                                    nkwargs['imgopts'])

        return nkwargs

    def parse_csv(
            self,
            fpath:str,
            csvopts:Dict
        ) -> pd.DataFrame:
        """Parses a CSV file and returns a pandas DataFrame.

        Args:
            fpath (str): The file path of the CSV file to be parsed.
            csvopts (Dict): A dictionary containing the options to be passed to the `pd.read_csv`
                            function.

        Returns:
            pd.DataFrame: The parsed CSV data as a pandas DataFrame.

        Raises:
            None.

        Note:
            - If the 'usecols' option is specified in `csvopts` and is an instance of
              `np.ndarray` or `list`, only the specified columns will be returned in the
              DataFrame.
            - If the 'usecols' option is not specified or is not an instance of `np.ndarray`
              or `list`, all columns will be returned in the DataFrame.
        """
        #TODO: add some extra exceptions ~ convert to numpy array perhaps
        print('parsing '+fpath)
        if 'usecols' in csvopts and isinstance(csvopts['usecols'], (np.ndarray, list)):
            return pd.read_csv(fpath, **csvopts)[csvopts['usecols']]
        else:
            return pd.read_csv(fpath, **csvopts)

    def parse_xls(
            self,
            fpath:str,
            xlsopts:Dict
        ) -> pd.DataFrame:
        """Parses an Excel file and returns a pandas DataFrame.

        Args:
            fpath (str): The file path of the Excel file to parse.
            xlsopts (Dict): A dictionary of options to pass to the `pd.read_excel` function.

        Returns:
            pd.DataFrame: The parsed Excel data as a pandas DataFrame.

        Raises:
            TypeError: If `xlsopts['usecols']` is not an instance of `np.ndarray` or `list`.

        Note:
            This function can handle both regular Excel files and Excel files with specific columns.

        Example:
            >>> parse_xls('data.xlsx', {'sheet_name': 'Sheet1', 'usecols': ['A', 'B', 'C']})
        """
        #TODO: add some extra exceptions ~ convert to numpy array perhaps
        print('parsing '+fpath)
        if 'usecols' in xlsopts and isinstance(xlsopts['usecols'], (np.ndarray, list)):
            return pd.read_excel(fpath, **xlsopts)[xlsopts['usecols']]
        else:
            return pd.read_excel(fpath, **xlsopts)

    def parse_img(
            self,
            fpath:str,
            imgopts:Dict
        ) -> np.ndarray:
        """Parses an image file and applies specified image options.

        Args:
            fpath (str): The file path of the image.
            imgopts (Dict): A dictionary containing image options.

        Returns:
            np.ndarray: The parsed image as a NumPy array.

        Raises:
            None.

        Example:
            parse_img('path/to/image.jpg', {'mode': 0, 'space': 2, 'resize': [100, 100]})
        """
        if 'mode' in imgopts and isinstance(imgopts['mode'],\
                                            (int, np.int8, np.int16, np.int32, np.int64)):
            mode = imgopts['mode']
        else:
            mode = 1
        if 'space' in imgopts and isinstance(imgopts['space'],\
                                             (int, np.int8, np.int16, np.int32, np.int64)):
            space = imgopts['space']
        else:
            space = 4
        icontent = cv2.imread(fpath, mode)
        icontent = cv2.cvtColor(icontent, space)
        if 'resize' in imgopts and isinstance(imgopts['resize'], (np.ndarray, list)):
            icontent = cv2.resize(icontent, tuple(imgopts['resize']))
        return icontent

    def prepare(
            self,
            **kwargs:Any
        ) -> Tuple:
        """Prepares the data for further processing.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple: A tuple containing the prepared data.

        Raises:
            None.

        Examples:
            >>> prepare(**kwargs)
            (prepared_data,)
        """
        nkwargs = locals()['kwargs']
        if 'prepare' in nkwargs and nkwargs['prepare'] and\
            'prepcmds' in nkwargs and len(nkwargs['prepcmds']) and\
            len(nkwargs['files']) and isinstance(nkwargs['files'][0]['content'], pd.DataFrame):
            for i in range(len(nkwargs['files'])):
                if isinstance(nkwargs['files'][i]['content'], pd.DataFrame):
                    df = nkwargs['files'][i]['content'].copy()
                    cmds = nkwargs['prepcmds'].copy()
                    cmds.insert(0, "df = dfo.copy(deep=True)")
                    cmds.insert(0, "def prep(dfo):")
                    cmds.append("return df")
                    exec("\r\n\t".join(cmds))
                    df = eval("prep(df)")
                    nkwargs['files'][i]['content'] = df.copy()
                    del df
        if len(nkwargs['files']) == 1:
            #TODO use gather and args to split and convert files
            return nkwargs['files'][0]['content']
        else:
            #splits = list(set(i['filesplit'] for i in nkwargs['files']))
            splits = nkwargs['filesplits']
            all_contents = []
            all_labels = []
            for split in splits:
                if 'target' in nkwargs:
                    if nkwargs['files'][0]['filetype'] == 'csv':
                        contents, labels = list(zip(*[(i['content'].drop(nkwargs['target'],axis=1),\
                                                       i['content'][nkwargs['target']])\
                                                      for i in nkwargs['files']\
                                                        if i["filesplit"] == split]))
                    else:
                        contents, labels = list(zip(*[(i['content'], i[nkwargs['target']])\
                                            for i in nkwargs['files'] if i["filesplit"] == split]))
                else:
                    contents = [i['content'] for i in nkwargs['files'] if i["filesplit"] == split]
                    labels = []
                if len(labels) and isinstance(labels, (tuple, list)):
                    labels = np.array(list(labels)).reshape(-1, 1)
                if len(contents) and isinstance(contents[0], (np.ndarray)):
                    contents = np.array(contents)
                    if 'prepare' in nkwargs and nkwargs['prepare'] and\
                        'prepcmds' in nkwargs and len(nkwargs['prepcmds']):
                        arr = np.copy(contents)
                        cmds = nkwargs['prepcmds']
                        #cmds.insert(0, "arr = np.copy(arro)")
                        cmds.insert(0, "def prep(arr):")
                        cmds.append("return arr")
                        exec("\n    ".join(cmds))
                        contents = eval("prep(arr)")
                        #contents = np.copy(arr)
                        del arr
                if isinstance(contents, (list, tuple)) and len(contents) == 1:
                    contents = contents[0]
                if isinstance(labels, (list, tuple)) and len(labels) == 1:
                    labels = labels[0]
                if len(contents) > 0:
                    all_contents.append(contents)
                if len(labels) > 0:
                    all_labels.append(labels)
            return tuple(all_contents + all_labels)

    def gather(
            self,
            files:list
        ) -> Any:
        """Gathers the given list of files and returns the result.

        Args:
            files (list): A list of files to be gathered.

        Returns:
            Any: The result of gathering the files.

        Todo:
            - Sort and join the files by group (filesplit, filetype).

        Example:
            >>> gather(['file1.txt', 'file2.txt'])
            'file1.txtfile2.txt'
        """
        #TODO sort and join by group (filesplit, filetype)
        #sorted(files, key=lambda d: (d['filesplit'], d['filetype'], d['__filepath__']))

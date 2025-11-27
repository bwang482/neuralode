import os
import math
import json
import random
import pickle
import numpy as np
import pandas as pd
from operator import itemgetter
import torch.distributed as dist
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the folder component of the filename does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_txt(obj, filename, create_folder=True):
    if create_folder:
        _create_folder_if_not_exist(filename)

    np.savetxt(filename, obj)
    

def save_pickle(obj, filename, use_dill=False, protocol=5, create_folder=True):
    """ Basic pickle/dill dumping.
    Given a python object and a filename, the method will save the object under that filename.
    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        use_dill (bool): Set True to save using dill.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.
    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(filename, use_dill=False):
    """ Basic dill/pickle load function.
    Args:
        filename (str): Location of the object.
        use_dill (bool): Set True to load with dill.
    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_json(obj, filename, create_folder=True):
    """ Save file with json. """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        json.dump(obj, file)


def load_json(filename):
    """ Load file with json. """
    with open(filename) as file:
        obj = json.load(file)
    return obj


def groupby_apply_parallel(grouped_df, func, *args):
    """ Performs a pandas groupby.apply operation in parallel.
    Args:
        grouped_df (grouped dataframe): A dataframe that has been grouped by some key.
        func (python function): A python function that can act on the grouped dataframe.
        *args (list): List of arguments to be supplied to the function.
    Returns:
        dataframe: The dataframe after application of the function.
    """
    with Pool(cpu_count()) as p:
        return_list = p.starmap(func, [(group, *args) for name, group in grouped_df])
    return pd.concat(return_list)
    

def is_nan(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ('nan', 'none', ''):
        return True
    return False


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]
    

def calc_date_diff(d1, d2):
    return (d1 - d2) / np.timedelta64(1, 'Y')


def combine_lists(dates, codes):
    dd = defaultdict(list)
    for k, v in zip(dates, codes):
        dd[k].append(v)
    combined = sorted([x for x in dd.items()], key=itemgetter(0))
    return combined


def list2str(a):
    return "-".join(str(x) for x in a)


def dt_setup(rank, world_size):
    """
    Setup distributed training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def dt_cleanup():
    """
    Clean up distributed training
    """
    dist.destroy_process_group()



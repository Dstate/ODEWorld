import os
import time
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image

def save_numpy_to_pil(np_img, img_name):
    imgpil = Image.fromarray(np_img)
    imgpil.save(img_name)

def print_item_details(data, depth=1):
    indentation = '   ' * depth    
    data_type = type(data).__name__
    print(f"{indentation}data_type: {data_type}")

    if hasattr(data, "shape"):
        print(f"{indentation}data_value: {data.shape}")
    elif hasattr(data, '__len__'):
        print(f"{indentation}data_value: {len(data)} * {type(data[0]).__name__}")
    else :
        print(f"{indentation}data_value: {data}")

def print_hdf5_details(group):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"Group: {key}")
            print_hdf5_details(item)  # 递归查看子组
        else:
            print(f"Dataset: {key} - Shape: {item.shape}, dtype: {item.dtype}")

def load_pickle(file_path):
    file_size = osp.getsize(file_path)
    start_time = time.time()
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    load_time = time.time() - start_time
    print(f"Loaded data successfully from {file_path}, "
          f"file size: {file_size / (1024 * 1024):.2f} MB, "
          f"load time: {load_time:.4f} seconds")
    return data

def save_pickle(data, file_path="data.pkl"):
    start_time = time.time()
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    save_time = time.time() - start_time
    file_size = osp.getsize(file_path)
    print(f"Saved data to {file_path}, "
          f"file size: {file_size / (1024 * 1024):.2f} MB, "
          f"save time: {save_time:.4f} seconds")

def insert_dict_to_pickle(data: dict, file_path):
    pickle_data = load_pickle(file_path)
    for k,v in data.items():
        pickle_data[k] = v
    save_pickle(pickle_data, file_path)

def check_dict_structure(data, depth=1):
    indentation = '---' * depth
    if not isinstance(data, dict):
        print_item_details(data, depth)
    else :
        for key, value in data.items():
            print(f"{indentation}{key}")
            check_dict_structure(value, depth+1)

def merge_dict_to_dict(source: dict, target: dict) -> dict:
    """
    Merge key-value pairs from source into target.
    If a key exists in target, it will be updated and a message will be printed.

    :param source: Dictionary with new values to merge.
    :param target: Dictionary to be updated.
    :return: The updated target dictionary.
    """
    for key, value in source.items():
        if key in target:
            print(f"Merging dict to dict: Key '{key}' exists, updated. {target[key]} -> {value}")
        target[key] = value
    return target

def check_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print_hdf5_details(f)

def check_pickle_structure(file_path):
    data = load_pickle(file_path)
    check_dict_structure(data)
"""SHIFT dataset for mmdet.

This is a reference code for mmdet style dataset of the SHIFT dataset. Note that
only single-view 2D detection, instance segmentation, and tracking are supported.
Please refer to the torch version of the dataloader for multi-view multi-task cases.

The codes are tested in mmdet-2.20.0.


Example
-------
Below is a snippet showing how to add the SHIFTDataset class in mmdet config files.

    >>> dict(
    >>>     type='SHIFTDataset',
    >>>     data_root='./SHIFT_dataset/discrete/images'
    >>>     ann_file='train/front/det_2d.json',
    >>>     img_prefix='train/front/img.zip',
    >>>     backend_type='zip',
    >>>     pipeline=[
    >>>        ...
    >>>     ]
    >>> )


Notes
-----
1.  Please copy this file to `mmdet/datasets/` and update the `mmdet/datasets/__init__.py`
    so that the `SHIFTDataset` class is imported. You can refer to their official tutorial at
    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html.
2.  The `backend_type` must be one of ['file', 'zip', 'hdf5'] and the `img_prefix`
    must be consistent with the backend_type.
3.  Since the images are loaded before the pipeline with the selected backend, there is no need
    to add a `LoadImageFromFile` module in the pipeline again.
4.  For instance segmentation please use the `det_insseg_2d.json` for the `ann_file`,
    and add a `LoadAnnotations(with_mask=True)` module in the pipeline.
"""
from __future__ import annotations

import json
import os
import sys

import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import LoadAnnotations

import os
from abc import abstractmethod
from zipfile import ZipFile

try:
    import h5py
    from h5py import File
except:
    raise ImportError("Please install h5py to enable HDF5Backend.")

import numpy as np


# Add the root directory of the project to the path. Remove the following two lines
# if you have installed shift_dev as a package.
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

class DataBackend:
    """Abstract class of storage backends.

    All backends need to implement three functions: get(), set() and exists().
    get() reads the file as a byte stream and set() writes a byte stream to a
    file. exists() checks if a certain filepath exists.
    """

    @abstractmethod
    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content at the given filepath.

        Args:
            filepath (str): The filepath to store the data at.
            content (bytes): The content to store as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        """Get the file content at the given filepath as bytes.

        Args:
            filepath (str): The filepath to retrieve the data from."

        Returns:
            bytes: The content of the file as bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): The filepath to check.

        Returns:
            bool: True if the filepath exists, False otherwise.
        """
        raise NotImplementedError


class FileBackend(DataBackend):
    """Raw file from hard disk data backend."""

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        return os.path.exists(filepath)

    def set(self, filepath: str, content: bytes) -> None:
        """Write the file content to disk.

        Args:
            filepath (str): Path to file.
            content (bytes): Content to write in bytes.
        """
        with open(filepath, "wb") as f:
            f.write(content)

    def get(self, filepath: str) -> bytes:
        """Get file content as bytes.

        Args:
            filepath (str): Path to file.

        Raises:
            FileNotFoundError: If filepath does not exist.

        Returns:
            bytes: File content as bytes.
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"File not found:" f" {filepath}")
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf


class HDF5Backend(DataBackend):
    """Backend for loading data from HDF5 files.

    This backend works with filepaths pointing to valid HDF5 files. We assume
    that the given HDF5 file contains the whole dataset associated to this
    backend.

    You can use the provided script at vis4d/data/datasets/to_hdf5.py to
    convert your dataset to the expected hdf5 format before using this backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, File] = {}

    @staticmethod
    def _get_hdf5_path(filepath: str) -> tuple[str, list[str]]:
        """Get .hdf5 path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.hdf5/key1/key2'

        Returns:
            tuple[str, list[str]]: The .hdf5 path and the keys to retrieve.
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".hdf5" and not h5py.is_hdf5(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .hdf5 file
            if not filepath.endswith(".hdf5"):
                filepath = filepath + ".hdf5"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)
        if not os.path.exists(hdf5_path):
            return False
        value_buf = self._get_client(hdf5_path, "r")

        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                return False
        return True

    def set(self, filepath: str, content: bytes) -> None:
        """Set the file content.

        Args:
            filepath: path/to/file.hdf5/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .hdf5 file
        """
        if ".hdf5" not in filepath:
            raise ValueError(f"{filepath} not a valid .hdf5 filepath!")
        hdf5_path, keys_str = filepath.split(".hdf5")
        key_list = keys_str.split("/")
        file = self._get_client(hdf5_path + ".hdf5", "a")
        if len(key_list) > 1:
            group_str = "/".join(key_list[:-1])
            if group_str == "":
                group_str = "/"

            group = file[group_str]
            key = key_list[-1]
            group.create_dataset(key, data=np.frombuffer(content, dtype="uint8"))

    def _get_client(self, hdf5_path: str, mode: str) -> File:
        """Get HDF5 client from path.

        Args:
            hdf5_path (str): Path to HDF5 file.
            mode (str): Mode to open the file in.

        Returns:
            File: the hdf5 file.
        """
        if hdf5_path not in self.db_cache:
            client = File(hdf5_path, mode)
            self.db_cache[hdf5_path] = [client, mode]
        else:
            client, current_mode = self.db_cache[hdf5_path]
            if current_mode != mode:
                client.close()
                client = File(hdf5_path, mode)
                self.db_cache[hdf5_path] = [client, mode]
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an HDF5 path
                together with the relative path inside it, e.g.: "/path/to/
                file.hdf5/key/subkey/data". If no .hdf5 given inside filepath,
                the function will search for the first .hdf5 file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.hdf5.

        Raises:
            FileNotFoundError: If no suitable file exists.
            ValueError: If key not found inside hdf5 file.

        Returns:
            bytes: The file content in bytes
        """
        hdf5_path, keys = self._get_hdf5_path(filepath)

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"Corresponding HDF5 file not found:" f" {filepath}"
            )
        value_buf = self._get_client(hdf5_path, "r")
        url = "/".join(reversed(keys))
        while keys:
            value_buf = value_buf.get(keys.pop())
            if value_buf is None:
                raise ValueError(f"Value {url} not found in {filepath}!")

        return bytes(value_buf[()])


class ZipBackend(DataBackend):
    """Backend for loading data from Zip files.

    This backend works with filepaths pointing to valid Zip files. We assume
    that the given Zip file contains the whole dataset associated to this
    backend.
    """

    def __init__(self) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.db_cache: dict[str, tuple[ZipFile, str]] = {}

    @staticmethod
    def _get_zip_path(filepath: str) -> tuple[str, list[str]]:
        """Get .zip path and keys from filepath.

        Args:
            filepath (str): The filepath to retrieve the data from.
                Should have the following format: 'path/to/file.zip/key1/key2'

        Returns:
            tuple[str, list[str]]: The .zip path and the keys to retrieve.
        """
        filepath_as_list = filepath.split("/")
        keys = []

        while filepath != ".zip" and not os.path.exists(filepath):
            keys.append(filepath_as_list.pop())
            filepath = "/".join(filepath_as_list)
            # in case data_root is not explicitly set to a .zip file
            if not filepath.endswith(".zip"):
                filepath = filepath + ".zip"
        return filepath, keys

    def exists(self, filepath: str) -> bool:
        """Check if filepath exists.

        Args:
            filepath (str): Path to file.

        Returns:
            bool: True if file exists, False otherwise.
        """
        zip_path, keys = self._get_zip_path(filepath)
        if not os.path.exists(zip_path):
            return False
        file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        return url in file.namelist()

    def set(self, filepath: str, content: bytes) -> None:
        """Write the file content to the zip file.

        Args:
            filepath: path/to/file.zip/key1/key2/key3
            content: Bytes to be written to entry key3 within group key2
            within another group key1, for example.

        Raises:
            ValueError: If filepath is not a valid .zip file
            NotImplementedError: If the method is not implemented.
        """
        if ".zip" not in filepath:
            raise ValueError(f"{filepath} not a valid .zip filepath!")

        zip_path, keys = self._get_zip_path(filepath)
        zip_file = self._get_client(zip_path, "a")
        url = "/".join(reversed(keys))
        zip_file.writestr(url, content)

    def _get_client(self, zip_path: str, mode: Literal["r", "w", "a", "x"]) -> ZipFile:
        """Get Zip client from path.

        Args:
            zip_path (str): Path to Zip file.
            mode (str): Mode to open the file in.

        Returns:
            ZipFile: the hdf5 file.
        """
        assert len(mode) == 1, "Mode must be a single character for zip file."
        if zip_path not in self.db_cache:
            client = ZipFile(zip_path, mode)
            self.db_cache[zip_path] = (client, mode)
        else:
            client, current_mode = self.db_cache[zip_path]
            if current_mode != mode:
                client.close()
                client = ZipFile(zip_path, mode)  # pylint:disable=consider-using-with
                self.db_cache[zip_path] = (client, mode)
        return client

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath as bytes.

        Args:
            filepath (str): The path to the file. It consists of an Zip path
                together with the relative path inside it, e.g.: "/path/to/
                file.zip/key/subkey/data". If no .zip given inside filepath,
                the function will search for the first .zip file present in
                the path, i.e. "/path/to/file/key/subkey/data" will also /key/
                subkey/data from /path/to/file.zip.

        Raises:
            ZipFileNotFoundError: If no suitable file exists.
            OSError: If the file cannot be opened.
            ValueError: If key not found inside zip file.

        Returns:
            bytes: The file content in bytes
        """
        zip_path, keys = self._get_zip_path(filepath)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Corresponding zip file not found:" f" {filepath}")
        zip_file = self._get_client(zip_path, "r")
        url = "/".join(reversed(keys))
        try:
            with zip_file.open(url) as zf:
                content = zf.read()
        except KeyError as e:
            raise ValueError(f"Value '{url}' not found in {zip_path}!") from e
        return bytes(content)


@DATASETS.register_module()
class ShiftDataset(CustomDataset):
    CLASSES = ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle")

    WIDTH = 1280
    HEIGHT = 800

    def __init__(self, *args, backend_type: str = "file", **kwargs):
        """Initialize the SHIFT dataset.

        Args:
            backend_type (str, optional): The type of the backend. Must be one of
                ['file', 'zip', 'hdf5']. Defaults to "file".
        """
        super().__init__(*args, **kwargs)
        self.backend_type = backend_type
        if backend_type == "file":
            self.backend = None
        elif backend_type == "zip":
            self.backend = ZipBackend()
        elif backend_type == "hdf5":
            self.backend = HDF5Backend()
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}! "
                "Must be one of ['file', 'zip', 'hdf5']"
            )

    def load_annotations(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)

        data_infos = []
        for img_info in data["frames"]:
            img_filename = os.path.join(
                self.img_prefix, img_info["videoName"], img_info["name"]
            )

            bboxes = []
            labels = []
            track_ids = []
            masks = []
            for label in img_info["labels"]:
                if label["category"] not in self.CLASSES:
                    continue
                bbox = label["box2d"]
                bboxes.append((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                labels.append(self.CLASSES.index(label["category"]))
                track_ids.append(label["id"])
                if "rle" in label and label["rle"] is not None:
                    masks.append(label["rle"])

            data_infos.append(
                dict(
                    filename=img_filename,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64),
                        track_ids=np.array(track_ids).astype(np.int64),
                        masks=masks if len(masks) > 0 else None,
                    ),
                )
            )
        return data_infos

    def get_img(self, idx):
        filename = self.data_infos[idx]["filename"]
        if self.backend_type == "zip":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        elif self.backend_type == "hdf5":
            img_bytes = self.backend.get(filename)
            return mmcv.imfrombytes(img_bytes)
        else:
            return mmcv.imread(filename)

    def get_img_info(self, idx):
        return dict(
            filename=self.data_infos[idx]["filename"],
            width=self.WIDTH,
            height=self.HEIGHT,
        )

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]

    def prepare_train_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        # Filter out images without annotations during training
        if len(ann_info["bboxes"]) == 0:
            return None
        results = dict(img=img, img_info=img_info, ann_info=ann_info, filename=img_info["filename"], ori_filename=img_info["filename"], img_shape=img.shape, ori_shape=img.shape, img_fields=['img'])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img = self.get_img(idx)
        img_info = self.get_img_info(idx)
        results = dict(img=img, img_info=img_info, filename=img_info["filename"], ori_filename=img_info["filename"], img_shape=img.shape, ori_shape=img.shape, img_fields=['img'])
        self.pre_pipeline(results)
        return self.pipeline(results)


if __name__ == "__main__":
    """Example for loading the SHIFT dataset for instance segmentation."""

    dataset = SHIFTDataset(
        data_root="./SHIFT_dataset/discrete/images",
        ann_file="train/front/det_insseg_2d.json",
        img_prefix="train/front/img.zip",
        backend_type="zip",
        pipeline=[LoadAnnotations(with_mask=True)],
    )

    # Print the dataset size
    print(f"Total number of samples: {len(dataset)}.")

    # Print the tensor shape of the first batch.
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        print("img:", data["img"].shape)
        print("ann_info.bboxes:", data["ann_info"]["bboxes"].shape)
        print("ann_info.labels:", data["ann_info"]["labels"].shape)
        print("ann_info.track_ids:", data["ann_info"]["track_ids"].shape)
        if "gt_masks" in data:
            print("gt_masks:", data["gt_masks"])
        break

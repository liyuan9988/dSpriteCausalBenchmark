import numpy as np
from numpy.random import default_rng
from itertools import product
from filelock import FileLock
import pathlib

from ..proxy_causal.data_class import PVTrainDataSet, PVTestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
             orientation_id_arr: np.ndarray,
             scale_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)


def structural_func(image, weights):
    return (image.dot(weights)[:, 0] ** 2 - 3000) / 500

def cal_weight():
    weights = np.empty((64, 64))
    for i in range(64):
        for j in range(64):
            weights[i, j] = (np.abs(32 - j))
    return weights.reshape(64*64, 1) / 32


def generate_test_dsprite() -> PVTestDataSet:
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")

    weights = cal_weight()

    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
    posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
    scale_id_arr = [0, 3, 5]
    orientation_arr = [0, 10, 20, 30]
    latent_idx_arr = []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

    image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
    data_size = 7 * 7 * 3 * 4
    treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
    structural = structural_func(treatment, weights)
    structural = structural[:, np.newaxis]
    return PVTestDataSet(treatment=treatment, structural=structural)


def generate_train_dsprite(n_sample: int,
                            rand_seed: int = 42, **kwargs):
    """
    Parameters
    ----------
    n_sample : int
        size of data
    rand_seed : int
        random seed
    """
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")


    weights = cal_weight()

    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    rng = default_rng(seed=rand_seed)
    posX_id_arr = rng.integers(32, size=n_sample)
    posY_id_arr = rng.integers(32, size=n_sample)
    scale_id_arr = rng.integers(6, size=n_sample)
    orientation_arr = rng.integers(40, size=n_sample)
    image_idx_arr = image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
    treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
    treatment += rng.normal(0.0, 0.1, treatment.shape)
    latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
    treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)

    posX_id_proxy = np.array([16] * n_sample)
    scale_id_proxy = np.array([3] * n_sample)
    orientation_proxy = np.array([0] * n_sample)
    proxy_image_idx_arr = image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
    outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
    outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

    structural = structural_func(treatment, weights)
    outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, n_sample)
    outcome = outcome[:, np.newaxis]

    train_data = PVTrainDataSet(treatment=treatment,
                                treatment_proxy=treatment_proxy,
                                outcome_proxy=outcome_proxy,
                                outcome=outcome)

    test_data = generate_test_dsprite()
    return train_data, test_data
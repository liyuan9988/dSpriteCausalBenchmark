import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from filelock import FileLock
from itertools import product
import pathlib

from src.frontdoor_backdoor.data_class import BackDoorTrainDataSet, ATETestDataSet, FrontDoorTrainDataSet, ATTTestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")


def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    orientation_id_arr = np.array([0] * data_size, dtype=int)
    scale_id_arr = np.array([0] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)


def obtain_images(obs_latent, imgs, latents_bases, length=1):
    points = np.linspace(0.0, 1.0, 32)[:, np.newaxis] * 3 - 1.5
    X_arr = np.argmin(cdist(points, obs_latent[:, [0]]), axis=0)
    Y_arr = np.argmin(cdist(points, obs_latent[:, [1]]), axis=0)
    image_idx_arr = image_id(latents_bases, X_arr, Y_arr)
    obs = imgs[image_idx_arr].reshape((length, 64 * 64)).astype(np.float32)
    return obs


def structural_func(image, weights):
    return (np.mean(image.dot(weights), axis=1)) ** 2 / 100.0


def load_dsprite():
    # compile dSprite
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")

    weights = (np.arange(64)[:, np.newaxis] / 64) @ (np.arange(64)[:, np.newaxis] / 64).T
    weights = weights.reshape(-1, 1)

    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))
    return imgs, weights, latents_bases


def generate_ate_dsprite_backdoor(n_data: int, rand_seed: int, **kwargs):
    rng = default_rng(seed=rand_seed)
    latent_r = rng.uniform(0.0, 1.0, n_data)
    latent_theta = rng.uniform(0.0, 2 * np.pi, n_data)
    backdoor = np.concatenate([(latent_r * np.cos(latent_theta))[:, np.newaxis],
                               (latent_r * np.sin(latent_theta))[:, np.newaxis]], axis=1)
    obs_noise = 0.3
    image_noise = 0.1
    backdoor = backdoor + rng.normal(0.0, obs_noise, (n_data, 2))

    imgs, weights, latents_bases = load_dsprite()
    obs = obtain_images(backdoor, imgs, latents_bases, n_data)
    obs += rng.normal(0.0, image_noise, obs.shape)
    outcome_noise = rng.normal(0.0, 0.5, n_data)
    outcome = structural_func(obs, weights) + 4 * (latent_r - 0.5) + outcome_noise

    train_data = BackDoorTrainDataSet(backdoor=backdoor,
                                      treatment=obs,
                                      outcome=outcome[:, np.newaxis])

    # Generate Test
    test_posX_id_arr = [0, 15, 31]
    test_posY_id_arr = [0, 15, 31]
    latent_idx_arr = []
    for posX, posY in product(test_posX_id_arr, test_posY_id_arr):
        latent_idx_arr.append([posX, posY])

    latent_idx_arr = np.array(latent_idx_arr)
    image_idx_arr = image_id(latents_bases, latent_idx_arr[:, 0], latent_idx_arr[:, 1])

    data_size = 9
    treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
    structural = structural_func(treatment, weights)
    structural = structural[:, np.newaxis]
    test_data = ATETestDataSet(treatment=treatment,
                               structural=structural)
    return train_data, test_data


def generate_att_dsprite_frontdoor(n_data: int, rand_seed: int, **kwargs) -> (FrontDoorTrainDataSet, ATTTestDataSet):
    rng = default_rng(seed=rand_seed)
    latent = rng.uniform(-1.5, 1.5, (n_data, 2))
    image_noise = 0.1
    front_door_noise = 0.2
    imgs, weights, latents_bases = load_dsprite()

    obs = obtain_images(latent, imgs, latents_bases, n_data)
    obs += rng.normal(0.0, image_noise, obs.shape)
    outcome_noise = 5 * (latent[:, 0] + latent[:, 1]) + rng.normal(0.0, 0.5, n_data)
    front_door = obs.dot(weights) + rng.normal(0.0, front_door_noise, (n_data, 1))
    outcome = front_door ** 2 / 100 + outcome_noise[:, np.newaxis]

    train_data = FrontDoorTrainDataSet(frontdoor=front_door,
                                       treatment=obs,
                                       outcome=outcome)

    n_test = 121
    treatment_ac_img_latent = np.array([[0.3, 0.3] for i in range(n_test)])
    treatment_ac = obtain_images(treatment_ac_img_latent, imgs, latents_bases, n_test)
    treatment_cf_img_latent = np.array([[posX, posY] for posX, posY in product(np.linspace(-1.0, 1.0, 11),
                                                                               np.linspace(-1.0, 1.0, 11))])
    treatment_cf = obtain_images(treatment_cf_img_latent, imgs, latents_bases, n_test)
    obs = obtain_images(treatment_cf, imgs, latents_bases, n_test)
    front_door_org = obs.dot(weights)
    target = np.zeros((n_test, 1))
    for i in range(100):
        front_door = front_door_org + rng.normal(0.0, front_door_noise, (n_test, 1))
        target = target + (front_door ** 2 / 100)
    target /= 100

    target = target + 5 * (treatment_ac_img_latent[:, [0]] + treatment_ac_img_latent[:, [1]])
    test_data = ATTTestDataSet(treatment_cf=treatment_cf,
                               treatment_ac=treatment_ac,
                               target=target)

    return train_data, test_data

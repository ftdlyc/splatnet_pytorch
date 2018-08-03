import numpy as np


def rotate_point_cloud(data):
    """

    :param data: Nx3 array
    :return: rotated_data: Nx3 array
    """
    angles = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angles)
    sinval = np.sin(angles)
    R = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    rotated_data = np.dot(data, R)
    return rotated_data


def random_rotate_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """

    :param data: Nx3 array
    :return: rotated_data: Nx3 array
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_data = np.dot(data, R)

    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """

    :param data: Nx3 array
    :return: jittered_data: Nx3 array
    """
    N, C = data.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += data

    return jittered_data


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    """

    :param data:  Nx3 array
    :return: scaled_data:  Nx3 array
    """
    scale = np.random.uniform(scale_low, scale_high)
    scaled_data = data * scale

    return scaled_data


def random_dropout_point_cloud(data, p=0.9):
    """

    :param data:  Nx3 array
    :return: dropout_data:  Nx3 array
    """
    N, C = data.shape
    dropout_ratio = np.random.random() * p
    drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
    dropout_data = np.zeros_like(data)
    if len(drop_idx) > 0:
        dropout_data[drop_idx, :] = data[0, :]

    return dropout_data


def shift_point_cloud(data, shift_range=0.1):
    """

    :param data:  Nx3 array
    :return: shift_data:  Nx3 array
    """
    N, C = data.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    shift_data = data + shifts
    return shift_data

import os
import pandas as pd
import SimpleITK as sitk
from lungmask import LMInferer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mutual_info_score
import math
import cv2
from skimage.transform import resize
from Data import center_crop
import torch


def lung_mask_extract(image_files_path, lung_mask_save_path, save_png_example=True, image_format='DICOM', image_direction='CT'):
    if image_format == "DICOM":
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(image_files_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(image_files_path)
    image_npy = sitk.GetArrayFromImage(image)

    spacing_npy = np.array(list(reversed(image.GetSpacing())))
    direction = image.GetDirection()

    # Check if a flip is needed based on the direction cosines
    transformM = np.array(direction).reshape(3, 3)
    transformM = np.round(transformM)
    isflip = np.any(transformM != np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inferer = LMInferer()
    inferer.device = device
    inferer.model.to(device)
    lung_mask = inferer.apply(image)

    binary_mask = (lung_mask == 1) | (lung_mask == 2)
    image_npy[~binary_mask] = 0

    if not os.path.exists(lung_mask_save_path):
        os.mkdir(lung_mask_save_path)

    if isflip:
        image_npy = image_npy[:, ::-1, ::-1]
    if save_png_example:
        fig_num = 5
        z_size = image_npy.shape[0]
        fig, ax = plt.subplots(fig_num, 1, figsize=(20, 10 * fig_num))
        for j in range(fig_num):
            ax[j].imshow(image_npy[int(z_size/2.0) + j, :, :], cmap='gray', vmin=-1200, vmax=600)
        plt.savefig(lung_mask_save_path + 'lung_mask_example.png')
        plt.close(fig)
    np.savez_compressed(lung_mask_save_path + 'image_lm.npz', image_npy)
    np.savez_compressed(lung_mask_save_path + 'spacing_lm.npz', spacing_npy)
    return 0


def gradient_richness(image, normalize=True):
    """
    计算图像的梯度并进行归一化

    参数:
    image: np.array, 2D 图像（灰度图）
    normalize: 是否归一化梯度值

    返回:
    float, 图像梯度的平均值和方差
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # 使用Sobel算子计算水平和垂直方向的梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化梯度幅值到 [0, 1]
    if normalize:
        grad_min = np.min(grad_magnitude)
        grad_max = np.max(grad_magnitude)
        if grad_max > grad_min:  # 避免除以零
            grad_magnitude = (grad_magnitude - grad_min) / (grad_max - grad_min)
        else:
            grad_magnitude = np.zeros_like(grad_magnitude)

    # 计算梯度的平均值和方差
    mean_gradient = np.mean(grad_magnitude)
    std_gradient = np.std(grad_magnitude)

    return mean_gradient, std_gradient


def CT_slice_selection(image, lung_mask_save_path, S=3, sw=5, show_result=False):
    '''
    image: 3维CT图像，Z, Y, X
    K: 最终选择的slice数量
    sw: slice垂直方向视窗宽度，单位为1
    mode: 评价图像差异度指标
    '''
    N = image.shape[0]
    back_sw_0 = math.floor(sw / 2.0)
    front_sw_0 = math.ceil(sw / 2.0)
    D_list = []
    grad_list = []
    for i in range(N):
        slice0 = image[i]
        grad, _ = gradient_richness(slice0)
        grad_list.append(grad)
        back_sw = back_sw_0
        front_sw = front_sw_0
        if i < back_sw_0:
            back_sw = i
        if i > N - front_sw_0 - 1:
            front_sw = N-i-1
        D = 0
        for j in range(i - back_sw, i + front_sw + 1):
            if j == i:
                continue
            slice1 = image[j]
            mi = mutual_info_score(slice0.ravel(), slice1.ravel())
            D += mi
        D_list.append(D/(back_sw + front_sw))
        # print(f'slice {i} D is {D}')
    max_D = max(D_list)
    D_list = [grad_list[i]*(max_D - d)/max_D for i, d in enumerate(D_list)]
    max_D_list = []
    max_index_list = []
    for s in range(S):
        split_index_start = s*int(N/S)
        split_index_end = (s+1)*int(N/S)
        if s == S-1:
            split_index_end = N-1
        max_D, max_index = max((val, idx) for idx, val in enumerate(D_list[split_index_start:split_index_end]))
        max_D_list.append(max_D)
        max_index_list.append(split_index_start + max_index)
    print('max_D_list :', max_D_list)
    print('max_index_list :', max_index_list)
    if not os.path.exists(lung_mask_save_path):
        os.mkdir(lung_mask_save_path)
    if show_result:
        fig_num = S
        fig, ax = plt.subplots(fig_num, 1, figsize=(20, 10 * fig_num))
        j = 0
        for index in max_index_list:
            ax[j].imshow(image[index, :, :], cmap='gray', vmin=-1200, vmax=600)
            j = j + 1
        plt.savefig(lung_mask_save_path + '/image_example.png')
        plt.close(fig)
    np.savez_compressed(lung_mask_save_path + '/image_lm.npz', image[max_index_list])


def uniform_CT_slice_selection(image, lung_mask_save_path, S=3):
    slice_num = image.shape[0]
    if slice_num > 50:
        cut_image = image[30:-20]
    else:
        raise ValueError("The slice number of CT is too small!")
    slice_num = cut_image.shape[0]
    sub_slice_num = slice_num // S
    indices = [sub_slice_num * i + sub_slice_num // 2 for i in range(S)]
    if not os.path.exists(lung_mask_save_path):
        os.mkdir(lung_mask_save_path)
    np.savez_compressed(lung_mask_save_path + '/image_lm.npz', cut_image[indices])


def CT_lung_mask_2_X_ray_20(CT_lung_mask_save_path, Xary_20_save_path, S=20, crop_size=(384, 384), target_shape=(256,256)):
    CT_lung_mask_image_npy = np.load(CT_lung_mask_save_path + 'image_lm.npz')['arr_0']
    slice_num = CT_lung_mask_image_npy.shape[1]
    if slice_num > 50:
        cut_image = CT_lung_mask_image_npy[:, 30:-20, :]
    else:
        raise ValueError("The slice number of CT is too small!")
    slice_num = cut_image.shape[1]
    sub_slice_num = slice_num // S
    indices = [sub_slice_num * i + sub_slice_num // 2 for i in range(S)]
    if not os.path.exists(Xary_20_save_path):
        os.mkdir(Xary_20_save_path)
    Xray_lung_mask_image_npy = cut_image[:, indices, :]
    Xray_lung_mask_image_npy = Xray_lung_mask_image_npy.transpose(1, 0, 2)
    Xray_lung_mask_image_npy = Xray_lung_mask_image_npy[:, ::-1, :]

    H = Xray_lung_mask_image_npy.shape[1]
    W = Xray_lung_mask_image_npy.shape[2]
    if H > crop_size[0] and W > crop_size[1]:
        cropped_image = center_crop(Xray_lung_mask_image_npy, crop_size=crop_size)
    else:
        cropped_image = Xray_lung_mask_image_npy
    cropped_image = cropped_image.astype(np.float32)
    resized_image = np.zeros((S, target_shape[0], target_shape[1]), dtype=np.float32)
    for i in range(cropped_image.shape[0]):
        resized_image[i] = cv2.resize(cropped_image[i], target_shape)

    np.savez_compressed(Xary_20_save_path + 'image_lm_cropped.npz', resized_image)

    fig_num = S
    fig, ax = plt.subplots(2, S // 2, figsize=(2 * S, 20))
    ax = ax.flatten()
    for j in range(fig_num):
        ax[j].imshow(resized_image[j], cmap='gray', vmin=-1200, vmax=600)
    plt.savefig(Xary_20_save_path + 'image_example.png')
    plt.close(fig)


def CT_2_X_ray_20(CT_save_path, Xary_20_save_path, S=20, crop_size=(512, 512), target_shape=(256, 256), CT_format='DICOM'):
    if os.path.exists(Xary_20_save_path + 'image_cropped.npz'):
        print('already exist.')
        try:
            image_data = np.load(Xary_20_save_path + 'image_cropped.npz')['arr_0'][5:15]
            return
        except Exception as e:
            print(f"[Error loading file] part{Xary_20_save_path + 'image_cropped.npz'}: {e}")

    if CT_format == "DICOM":
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(CT_save_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(CT_save_path)
    image_npy = sitk.GetArrayFromImage(image)

    spacing_npy = np.array(list(reversed(image.GetSpacing())))
    direction = image.GetDirection()

    # Check if a flip is needed based on the direction cosines
    transformM = np.array(direction).reshape(3, 3)
    transformM = np.round(transformM)
    isflip = np.any(transformM != np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    if isflip:
        image_npy = image_npy[:, ::-1, ::-1]

    slice_num = image_npy.shape[1]
    if slice_num > 50:
        cut_image = image_npy[:, 30:-20, :]
    else:
        raise ValueError("The slice number of CT is too small!")
    slice_num = cut_image.shape[1]
    sub_slice_num = slice_num // S
    indices = [sub_slice_num * i + sub_slice_num // 2 for i in range(S)]
    if not os.path.exists(Xary_20_save_path):
        os.mkdir(Xary_20_save_path)
    Xray_image_npy = cut_image[:, indices, :]
    Xray_image_npy = Xray_image_npy.transpose(1, 0, 2)
    Xray_image_npy = Xray_image_npy[:, ::-1, :]

    H = Xray_image_npy.shape[1]
    W = Xray_image_npy.shape[2]
    if H > crop_size[0] and W > crop_size[1]:
        cropped_image = center_crop(Xray_image_npy, crop_size=crop_size)
    else:
        cropped_image = Xray_image_npy
    cropped_image = cropped_image.astype(np.float32)
    resized_image = np.zeros((S, target_shape[0], target_shape[1]), dtype=np.float32)
    for i in range(cropped_image.shape[0]):
        resized_image[i] = cv2.resize(cropped_image[i], target_shape)

    np.savez_compressed(Xary_20_save_path + 'image_cropped.npz', resized_image)

    fig_num = S
    fig, ax = plt.subplots(2, S // 2, figsize=(2 * S, 20))
    ax = ax.flatten()
    for j in range(fig_num):
        ax[j].imshow(resized_image[j], cmap='gray', vmin=-1200, vmax=600)
    plt.savefig(Xary_20_save_path + 'image_example.png')
    plt.close('all')
    plt.clf()


if __name__ == '__main__':

    '''
    CT_root_paths = ['',
                     '',
                     '',
                     '',
                     '']

    X_ray_lungmask_root_save_paths = ['',
                             '',
                             '',
                             '',
                             '']

    X_ray_root_save_paths = ['',
                             '',
                             '',
                             '',
                             '']
    
    for i, CT_root_path in enumerate(CT_root_paths):
        X_ray_lungmask_root_save_path = X_ray_lungmask_root_save_paths[i]
        X_ray_root_save_path = X_ray_root_save_paths[i]
        ids = os.listdir(X_ray_lungmask_root_save_path)
        ids = [id for id in ids if 'txt' not in id]
        if i == 0:
            CT_format = 'DICOM'
        else:
            CT_format = 'NII'
        for id in ids:
            if i == 0:
                CT_path = CT_root_path + id + '/'
            else:
                CT_path = CT_root_path + id + '.nii.gz'
            Xray_save_path = X_ray_root_save_path + id + '/'
            CT_2_X_ray_20(CT_path, Xray_save_path, S=20, crop_size=(512, 512), target_shape=(256, 256),
                          CT_format=CT_format)

    '''
    CT_root_paths = ['']

    X_ray_root_save_paths = ['']

    for i, CT_root_path in enumerate(CT_root_paths):
        X_ray_root_save_path = X_ray_root_save_paths[i]
        file_names = os.listdir(CT_root_path)
        file_names = [fn for fn in file_names if fn.endswith('.nii.gz')]
        id_dates = [fn.split('.')[0] for fn in file_names]

        CT_format = 'NII'
        for id_date in id_dates:
            CT_path = CT_root_path + id_date + '.nii.gz'
            Xray_save_path = X_ray_root_save_path + id_date + '/'
            CT_2_X_ray_20(CT_path, Xray_save_path, S=20, crop_size=(512, 512), target_shape=(256, 256),
                          CT_format=CT_format)

    # os.system(
    #     "export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")

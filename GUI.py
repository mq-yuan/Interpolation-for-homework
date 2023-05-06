import os
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def select_file(select_path):
    # 单个文件选择
    selected_file_path = filedialog.askopenfilename()  # 使用askopenfilename函数选择单个文件
    select_path.set(selected_file_path)

def get_upscale_mask(upscale_ratio, img_shape):
    mask = np.zeros(img_shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i%upscale_ratio==0 and j%upscale_ratio==0: mask[i, j] = 255
    return mask

def Interpolator(img_path, mask_path, random_ratio, upscale_ratio, flag):
    img = cv2.imread(img_path)

    if flag == 1:
        mask_path = f'./Randomly remove {random_ratio}%.bmp'
        mask = np.ones(img.shape[:2]) * 255
        all_point = np.transpose(np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))).reshape((-1, 2))
        res = np.random.choice(all_point.shape[0], size=(int(all_point.shape[0] * (random_ratio/100))), replace=False)
        mask[tuple(np.transpose(all_point[res]))] = 0
    elif flag == 2:
        img_shape = (img.shape[0]*upscale_ratio, img.shape[1]*upscale_ratio, img.shape[2])
        mask = get_upscale_mask(upscale_ratio, img_shape[:2])
        mask_path = f"./Upscale image {upscale_ratio}.bmp"
    else:
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

    location = np.transpose(np.nonzero(mask))  # mask白色区域
    x = np.transpose(np.where(mask == 0))  # mask黑色区域

    # 插值
    if flag == 2:
        data = img.reshape(-1, img.shape[-1])
        y_RBF_TPS = RBFInterpolator(location, data, neighbors=100, kernel='thin_plate_spline')(x)  # RBF TPS
        masked = np.zeros(img_shape)
        masked[np.nonzero(mask)] = data

        ans_RBF_TPS = np.zeros(img_shape)
        ans_RBF_TPS[np.nonzero(mask)] = data
        ans_RBF_TPS[np.where(mask == 0)] = y_RBF_TPS
    else:
        data = img[np.nonzero(mask)]  # mask白色区域对应的img像素值
        y_RBF_GAUSSIAN = RBFInterpolator(location, data, neighbors=100, kernel='gaussian', epsilon=1)(x)  # RBF Gaussian
        y_RBF_TPS = RBFInterpolator(location, data, neighbors=100, kernel='thin_plate_spline')(x)  # RBF TPS
        y_NNI = NearestNDInterpolator(location, data)(x)  # Nearest neighbor interpolation
        y_BNI = LinearNDInterpolator(location, data)(x)  # Bilinear interpolation

        # 拼接
        masked = np.zeros(img.shape)
        masked[np.nonzero(mask)] = data

        ans_RBF_GAUSSIAN = np.zeros(img.shape)
        ans_RBF_GAUSSIAN[np.nonzero(mask)] = data
        ans_RBF_GAUSSIAN[np.where(mask == 0)] = y_RBF_GAUSSIAN

        ans_RBF_TPS = np.zeros(img.shape)
        ans_RBF_TPS[np.nonzero(mask)] = data
        ans_RBF_TPS[np.where(mask == 0)] = y_RBF_TPS

        ans_NNI = np.zeros(img.shape)
        ans_NNI[np.nonzero(mask)] = data
        ans_NNI[np.where(mask == 0)] = y_NNI

        ans_BNI = np.zeros(img.shape)
        ans_BNI[np.nonzero(mask)] = data
        ans_BNI[np.where(mask == 0)] = y_BNI

    # 保存
    img_name = os.path.splitext(os.path.split(img_path)[1])[0]
    mask_name = os.path.splitext(os.path.split(mask_path)[1])[0]
    img_dir = os.path.join('./Ans/', img_name)
    mask_dir = os.path.join(img_dir, mask_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if flag == 2:
        cv2.imwrite(os.path.join(mask_dir, 'Orignal.bmp'), img)
        cv2.imwrite(os.path.join(mask_dir, 'Masked.bmp'), masked)
        cv2.imwrite(os.path.join(mask_dir, 'RBF_TPS.bmp'), ans_RBF_TPS)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8),
                             sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Original image')
        img = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'Orignal.bmp')), cv2.COLOR_BGR2RGB)
        ax[0].imshow(img)

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('Masked image')
        masked = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'Masked.bmp')), cv2.COLOR_BGR2RGB)
        ax[1].imshow(masked)

        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('RBF thin_plate_spline')
        ans_RBF_TPS = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'RBF_TPS.bmp')), cv2.COLOR_BGR2RGB)
        ax[2].imshow(ans_RBF_TPS)
    else:
        cv2.imwrite(os.path.join(mask_dir, 'Orignal.bmp'), img)
        cv2.imwrite(os.path.join(mask_dir, 'Masked.bmp'), masked)
        cv2.imwrite(os.path.join(mask_dir, 'RBF_GAUSSIAN.bmp'), ans_RBF_GAUSSIAN)
        cv2.imwrite(os.path.join(mask_dir, 'RBF_TPS.bmp'), ans_RBF_TPS)
        cv2.imwrite(os.path.join(mask_dir, 'NNI.bmp'), ans_NNI)
        cv2.imwrite(os.path.join(mask_dir, 'BNI.bmp'), ans_BNI)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8),
                                sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., img / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., img / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., img / 255., data_range=1, channel_axis=2):.3f}')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Original image')
        img = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'Orignal.bmp')), cv2.COLOR_BGR2RGB)
        ax[0].imshow(img)

        ax[1].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., ans_NNI / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., ans_NNI / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., ans_NNI / 255., data_range=1, channel_axis=2):.3f}')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('NearestNDInterpolator')
        ans_NNI = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'NNI.bmp')), cv2.COLOR_BGR2RGB)
        ax[1].imshow(ans_NNI)

        ax[2].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., ans_BNI / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., ans_BNI / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., ans_BNI / 255., data_range=1, channel_axis=2):.3f}')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('Bilinear interpolation')
        ans_BNI = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'BNI.bmp')), cv2.COLOR_BGR2RGB)
        ax[2].imshow(ans_BNI)

        ax[3].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., masked / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., masked / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., masked / 255., data_range=1, channel_axis=2):.3f}')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_title('Masked image')
        masked = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'Masked.bmp')), cv2.COLOR_BGR2RGB)
        ax[3].imshow(masked)

        ax[4].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., ans_RBF_TPS / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., ans_RBF_TPS / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., ans_RBF_TPS / 255., data_range=1, channel_axis=2):.3f}')
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_title('RBF thin_plate_spline')
        ans_RBF_TPS = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'RBF_TPS.bmp')), cv2.COLOR_BGR2RGB)
        ax[4].imshow(ans_RBF_TPS)

        ax[5].set_xlabel(
            f'MSE: {mean_squared_error(img / 1., ans_RBF_GAUSSIAN / 1.):.2f}, PSNR: {peak_signal_noise_ratio(img / 255., ans_RBF_GAUSSIAN / 255., data_range=1):.2f}, SSIM: {structural_similarity(img / 255., ans_RBF_GAUSSIAN / 255., data_range=1, channel_axis=2):.3f}')
        ax[5].set_xticks([])
        ax[5].set_yticks([])
        ax[5].set_title('RBF Gaussian')
        ans_RBF_GAUSSIAN = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, 'RBF_GAUSSIAN.bmp')), cv2.COLOR_BGR2RGB)
        ax[5].imshow(ans_RBF_GAUSSIAN)

    plt.tight_layout()
    plt.savefig(os.path.join(mask_dir, 'Ans.png'), dpi=400)
    plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    img_path = tk.StringVar()
    mask_path = tk.StringVar()
    random_remove = tk.IntVar()
    upscale_ratio = tk.IntVar()
    var = tk.IntVar()
    random_remove.set(10)
    upscale_ratio.set(1)

    tk.Label(root, text="img path: ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=img_path).pack(side='top', anchor='n', fill='x')
    tk.Button(root, text="select img", command=lambda: select_file(img_path)).pack(side='top', anchor='ne')

    tk.Label(root, text="mask path: ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=mask_path).pack(side='top', anchor='n', fill='x')
    tk.Button(root, text="select mask", command=lambda: select_file(mask_path)).pack(side='top', anchor='ne')

    tk.Label(root, text="randomly remove ratio(int %): ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=random_remove).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="Upscale ratio(int): ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=upscale_ratio).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="Select the mode in which the image is destroyed: ").pack(side='top', anchor='nw')
    tk.Radiobutton(root, text="Mask image", variable=var, value=0, command=None).pack(side='top', anchor='nw')
    tk.Radiobutton(root, text="Randomly remove ", variable=var, value=1, command=None).pack(side='top', anchor='nw')
    tk.Radiobutton(root, text="Upscale image", variable=var, value=2, command=None).pack(side='top', anchor='nw')

    tk.Button(root, text="Interpolator", command=lambda: Interpolator(img_path.get(), mask_path.get(), random_remove.get(), upscale_ratio.get(), var.get())).pack()

    root.mainloop()

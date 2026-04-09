import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import re
from utils.util import worker_init_fn
from functools import partial
import scipy.ndimage
import torchvision.transforms.functional as TF
import gc


class CustomDataset(Dataset):
    def __init__(self, covariates, responses, mp_data, images=None, ct_reports=None, reasonings=None, reasonings_mask=None, image_augment=False, angle=0, scale_range=0):
        self.images = images
        self.covariates = covariates
        self.responses = responses
        self.image_augment = image_augment

        self.angle = angle
        self.scale_range = scale_range

        self.ct_reports = ct_reports

        self.reasonings = reasonings
        self.reasonings_mask = reasonings_mask

        self.mp_data = mp_data

    def __len__(self):
        return len(self.covariates)

    def __getitem__(self, idx):
        covariate = self.covariates[idx]
        response = self.responses[idx]
        if self.images is not None:
            image = self.images[idx]
            images = []
            if self.angle != 0:
                aug_angle = np.random.uniform(-self.angle, self.angle)
            else:
                aug_angle = 0
            if self.scale_range != 0:
                aug_scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
            else:
                aug_scale = 1.0
            translate = (0, 0)
            shear = 0

            for i in range(image.shape[0]):
                single_image = image[i]
                single_image = TF.to_tensor(single_image)
                if self.image_augment:
                    # single_image = self.image_transform(single_image)
                    single_image = TF.affine(
                        single_image,
                        angle=aug_angle,
                        translate=translate,
                        scale=aug_scale,
                        shear=shear,
                        fill=0
                    )
                images.append(single_image)
            image = torch.stack(images).squeeze()
            image = image.to(torch.float32)
        else:
            image = torch.tensor([], dtype=torch.float32)

        if self.ct_reports is not None:
            report = self.ct_reports[idx]
            report = torch.tensor(report, dtype=torch.float32)
        else:
            report = torch.tensor([], dtype=torch.float32)

        if self.reasonings is not None and self.reasonings_mask is not None:
            reasoning = self.reasonings[idx]
            reasoning_mask = self.reasonings_mask[idx]
            reasoning = torch.tensor(reasoning, dtype=torch.float32)
            reasoning_mask = torch.tensor(reasoning_mask, dtype=torch.float32)
        else:
            reasoning = torch.tensor([], dtype=torch.float32)
            reasoning_mask = torch.tensor([], dtype=torch.float32)

        diag = self.mp_data[idx]
        diag = torch.tensor(diag, dtype=torch.float32)

        return image, covariate, report, reasoning, reasoning_mask, response, diag


def center_crop(image, crop_size=(384, 384)):
    n, h, w = image.shape
    new_h, new_w = crop_size

    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2

    cropped_image = image[:, start_h:start_h + new_h, start_w:start_w + new_w]
    return cropped_image


def flip_images(images):
    flip_code = np.random.choice([0, 1, -1])
    if flip_code == 0:
        return images[:, ::-1, :]
    elif flip_code == 1:
        return images[:, :, ::-1]
    elif flip_code == -1:
        return images[:, ::-1, ::-1]
    return images


def rotate_images(images, k=None):
    if k is None:
        k = np.random.choice([0, 1, 2, 3])
    return np.rot90(images, k, axes=(1, 2))


def zoom_images(images, scale=None):
    if scale is None:
        scale = np.random.uniform(0.8, 1.2)

    n, h, w = images.shape
    new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
    zoomed = np.zeros_like(images)

    for i in range(n):
        zoomed_img = scipy.ndimage.zoom(images[i], zoom=(scale, scale), order=1)
        if scale > 1.0:
            crop_h, crop_w = (new_h - h) // 2, (new_w - w) // 2
            zoomed[i] = zoomed_img[crop_h:crop_h + h, crop_w:crop_w + w]
        else:
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            zoomed[i, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = zoomed_img

    return zoomed


def translate_images(images, max_shift=10):
    n, h, w = images.shape
    tx, ty = np.random.randint(-max_shift, max_shift + 1, size=2)
    translated = np.zeros_like(images)

    x_start, x_end = max(0, tx), min(w, w + tx)
    y_start, y_end = max(0, ty), min(h, h + ty)

    for i in range(n):
        translated[i, y_start:y_end, x_start:x_end] = images[i, max(0, -ty):h - max(0, ty), max(0, -tx):w - max(0, tx)]

    return translated


def augment_images(images_list, zoom_scale, rotate_k):
    augment_images_list = []
    for i in range(images_list.shape[0]):
        # augment_im = flip_images(images_list[i])
        augment_im = rotate_images(images_list[i], k=rotate_k)
        augment_im = zoom_images(augment_im, scale=zoom_scale)
        augment_im = translate_images(augment_im)
        # show_all_images(augment_im)
        augment_images_list.append(augment_im)
    augment_images_list = np.array(augment_images_list)
    return augment_images_list


def augment_image(image, zoom_scale, rotate_k):

    augment_im = rotate_images(image, k=rotate_k)
    if zoom_scale != 1.0:
        augment_im = zoom_images(augment_im, scale=zoom_scale)
    augment_im = translate_images(augment_im)

    return augment_im


def show_all_images(image_data):
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()
    num_images = len(image_data)
    rows, cols = 2, 10
    plt.figure(figsize=(20, 5))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_data[i], cmap="gray")
        plt.axis("off")
        plt.title(f"Img {i}")

    plt.tight_layout()
    plt.show()


def PFT_data_predictive_value(sex, height, age, PFT_metric='FEV1'):
    if PFT_metric not in ['FEV1', 'FVC', 'TLC', 'DLCO', 'VCMAX']:
        raise Exception("Input metric is not considered.")
    if PFT_metric == 'FEV1':
        biasm = -2.49
        hm_cc = 0.043
        am_cc = -0.029

        biasf = -2.60
        hf_cc = 0.0359
        af_cc = -0.025
    elif PFT_metric == 'FVC':
        biasm = -4.34
        hm_cc = 0.0576
        am_cc = -0.026

        biasf = -2.89
        hf_cc = 0.0443
        af_cc = -0.026
    elif PFT_metric == 'TLC':
        biasm = -7.08
        hm_cc = 0.0799
        am_cc = 0.0

        biasf = -5.79
        hf_cc = 0.066
        af_cc = 0.0
    elif PFT_metric == 'VCMAX':
        biasm = -4.65
        hm_cc = 0.061
        am_cc = -0.028

        biasf = -3.28
        hf_cc = 0.0466
        af_cc = -0.024
    else: # DLCOSB
        biasm = -6.03
        hm_cc = 0.1111
        am_cc = -0.066

        biasf = -2.74
        hf_cc = 0.0818
        af_cc = -0.049
    pv = hm_cc * height + am_cc * age + biasm if sex==0 else hf_cc * height + af_cc * age + biasf
    return float(pv)


def patient_train_val_split(PFT_paths, test_size=0.2, random_state=1, train_val_marks=['YD', 'P', 'MM']):
    df_list = [pd.read_excel(PFT_path).copy() for PFT_path in PFT_paths]
    train_val_ids = []
    for i, pft_df in enumerate(df_list):
        if i == 0:
            id_col_name = 'new_ID'
        else:
            id_col_name = 'Patient_ID'
        if i == 0:
            pft_df[id_col_name] = pft_df[id_col_name].astype(str)
            train_val_ids += pft_df[id_col_name].tolist()
        else:
            id_list = pft_df[id_col_name].tolist()
            tv_id_list = [id for id in id_list if any(k in id for k in train_val_marks)]
            train_val_ids += tv_id_list

    train_ids, val_ids = train_test_split(train_val_ids, test_size=test_size, random_state=random_state)
    return list(set(train_ids)), list(set(val_ids))


def process_image_cov_resp_report(image_folder, excel_file, IDs_train, IDs_val, image_flag=True, report_flag=False, mp_flag=False, report_path=None, LLM_predict=False, LLM_reasoning=False, LLM_reasoning_path=None, LLM_predict_path=None, log_file_path='data_process_log_file.txt', test_set_names=['CT', 'CJJ']):
    marks = ['YD', 'MM', 'CT', 'CJJ', 'YX']
    test_set_name1 = test_set_names[0]
    test_set_name2 = test_set_names[1]
    # 读取Excel中的数据
    df12 = pd.read_excel(excel_file[0]).copy()
    df3 = pd.read_excel(excel_file[1]).copy()
    df4 = pd.read_excel(excel_file[2]).copy()
    df5 = pd.read_excel(excel_file[3]).copy()
    df6 = pd.read_excel(excel_file[4]).copy()
    df7 = pd.read_excel(excel_file[5]).copy()
    df8 = pd.read_excel(excel_file[6]).copy()
    df9 = pd.read_excel(excel_file[7]).copy()
    df_list = [df12, df3, df4, df5, df6, df7, df8, df9]
    id_black_list = ['', '', '', '']

    LLM_df = pd.read_csv(LLM_predict_path[0], encoding='utf-8').copy()
    LLM_df['new_ID'] = LLM_df['new_ID'].astype(str)
    LLM_df3 = pd.read_csv(LLM_predict_path[1], encoding='utf-8').copy()
    LLM_df3['Patient_ID'] = LLM_df3['Patient_ID'].astype(str)
    LLM_df4 = pd.read_csv(LLM_predict_path[2], encoding='utf-8').copy()
    LLM_df4['Patient_ID'] = LLM_df4['Patient_ID'].astype(str)
    LLM_df5 = pd.read_csv(LLM_predict_path[3], encoding='utf-8').copy()
    LLM_df5['Patient_ID'] = LLM_df5['Patient_ID'].astype(str)
    LLM_df6 = pd.read_csv(LLM_predict_path[4], encoding='utf-8').copy()
    LLM_df6['Patient_ID'] = LLM_df6['Patient_ID'].astype(str)
    LLM_df7 = pd.read_csv(LLM_predict_path[5], encoding='utf-8').copy()
    LLM_df7['Patient_ID'] = LLM_df7['Patient_ID'].astype(str)
    LLM_df8 = pd.read_csv(LLM_predict_path[6], encoding='utf-8').copy()
    LLM_df8['Patient_ID'] = LLM_df8['Patient_ID'].astype(str)

    LLM_df9 = pd.read_csv(LLM_predict_path[7], encoding='utf-8').copy()
    LLM_df9['Patient_ID'] = LLM_df9['Patient_ID'].astype(str)

    image_folder_iddates_list = [os.listdir(img_f) for img_f in image_folder]
    ns_image_folder_iddates_list = []
    for i, image_folder_iddates in enumerate(image_folder_iddates_list):
        if i == 0 or i == 6 or i == 7:
            ns_image_folder_iddates_list.append(image_folder_iddates_list[i])
        else:
            # transform desensitized ids to initial format or inverse
            ns_image_folder_iddates = [iddate if 'M' not in iddate else 'MM' + iddate[4:] for iddate in image_folder_iddates]
            ns_image_folder_iddates = [iddate if any(k in iddate for k in marks) else 'YD11' + iddate[4:] for iddate in ns_image_folder_iddates]
            ns_image_folder_iddates_list.append(ns_image_folder_iddates)

    def pad_id(pid):
        if not any(x in pid for x in ['CJJ', 'CT', 'P', 'M', 'YD', 'YX']):
            return pid.zfill(10)
        return pid
    LLM_df3['Patient_ID'] = LLM_df3['Patient_ID'].apply(pad_id)
    LLM_df4['Patient_ID'] = LLM_df4['Patient_ID'].apply(pad_id)
    LLM_df5['Patient_ID'] = LLM_df5['Patient_ID'].apply(pad_id)
    LLM_df6['Patient_ID'] = LLM_df6['Patient_ID'].apply(pad_id)
    LLM_df7['Patient_ID'] = LLM_df7['Patient_ID'].apply(pad_id)
    LLM_df8['Patient_ID'] = LLM_df8['Patient_ID'].apply(pad_id)
    LLM_df9['Patient_ID'] = LLM_df9['Patient_ID'].apply(pad_id)

    LLM_df_list = [LLM_df, LLM_df3, LLM_df4, LLM_df5, LLM_df6, LLM_df7, LLM_df8, LLM_df9]

    ids_train, dates_train, images_train, covariates_train, responses_train, responses_mp_train, reports_train, diagnosises_train, reasoning_train, reasoning_mask_train = [], [], [], [], [], [], [], [], [], []
    ids_val, dates_val, images_val, covariates_val, responses_val, responses_mp_val, reports_val, diagnosises_val, reasoning_val, reasoning_mask_val = [], [], [], [], [], [], [], [], [], []
    ids_test_1, dates_test_1, images_test_1, covariates_test_1, responses_test_1, responses_mp_test_1, reports_test_1, diagnosises_test_1, reasoning_test_1, reasoning_mask_test_1 = [], [], [], [], [], [], [], [], [], []
    ids_test_2, dates_test_2, images_test_2, covariates_test_2, responses_test_2, responses_mp_test_2, reports_test_2, diagnosises_test_2, reasoning_test_2, reasoning_mask_test_2 = [], [], [], [], [], [], [], [], [], []
    FEV1FVC_pos_num = 0
    FEV1_pos_num = 0
    TLC_pos_num = 0
    DLCO_pos_num = 0
    for i, df in enumerate(df_list):
        if i == 0:
            id_col_name = 'new_ID'
            date_col_name = ''
            gender_col_name = '性别'
            age_col_name = '年龄'
            weight_col_name = '体重'
            height_col_name = '身高'
            des_col_name = 'CT所见'
            diag_col_name = 'CT诊断'
            FEV1_col_name = 'FEV1实测值'
            FEV1_pre_col_name = 'FEV1实测值/预计值'
            FVC_col_name = 'FVC实测值'
            FVC_pre_col_name = 'FVC实测值/预计值'
            TLC_col_name = 'TLC实测值'
            TLC_pre_col_name = 'TLC实测值/预计值'
            DLCO_col_name = 'DLCOcSB实测值'
            DLCO_pre_col_name = 'DLCOcSB实测值/预计值'
        elif i == 7:
            id_col_name = 'Patient_ID'
            date_col_name = ''
            gender_col_name = 'gender'
            age_col_name = 'age'
            weight_col_name = 'weight'
            height_col_name = 'height'
            des_col_name = 'Description'
            diag_col_name = 'Diagnosis'
            FEV1_col_worst_name = 'FEV1_最差'
            FEV1_col_name = 'FEV1_实测值'
            FEV1_pre_col_name = 'FEV1_预计值'
            FVC_col_worst_name = 'FVC_最差'
            FVC_col_name = 'FVC_实测值'
            # FVC_col_name = 'FVC实测值'
            FVC_pre_col_name = 'FVC_预计值'
            TLC_col_name = ''
            TLC_pre_col_name = 'TLC实测值/预计值'
            DLCO_col_name = 'DLCOcSB实测值'
            DLCO_pre_col_name = 'DLCOcSB实测值/预计值'
            VCMAX_col_name = 'Vcmax_实测值'
            VCMAX_pre_col_name = 'Vcmax_预计值'
        else:
            id_col_name = 'Patient_ID'
            date_col_name = 'Date_CTreport'
            gender_col_name = 'gender'
            age_col_name = 'age'
            weight_col_name = 'weight'
            height_col_name = 'height'
            des_col_name = 'description'
            diag_col_name = 'diagnosis'
            FEV1_col_name = 'FEV 1_实测值'
            FEV1_pre_col_name = 'FEV 1_预计值'
            FVC_col_name = 'FVC_实测值'
            FVC_pre_col_name = 'FVC_预计值'
            TLC_col_name = 'TLC_实测值'
            TLC_pre_col_name = 'TLC_预计值'
            DLCO_col_name = 'DLCOc SB_实测值'
            DLCO_pre_col_name = 'DLCOc SB_预计值'

        for k, row in df.iterrows():
            ID = str(row[id_col_name])
            '''
            if i != 0:
                if not 'P' in ID and not 'CT' in ID and not 'CJJ' in ID and not 'M' in ID:
                    ID = ID.zfill(10)
            '''
            if ID in id_black_list:
                continue
            # if ID not in IDs_train and ID not in IDs_val and not 'CT' in ID and not 'CJJ' in ID:
            # if ID not in IDs_train and ID not in IDs_val and not 'CT' in ID and not 'MM' in ID:
            if ID not in IDs_train and ID not in IDs_val and not any(tsn in ID for tsn in test_set_names):
                continue
            if i == 0 or i == 7:
                date = ''
            else:
                date = str(row[date_col_name])
                if i == 5:
                    date = ''.join(date.split(' ')[0].split('-'))
            # 20260216 update
            if i == 0 or i == 7:
                file_path = os.path.join(image_folder[i], f'{ID}', 'image_cropped.npz')
                ori_ID = ID
            elif i == 6:
                file_path = os.path.join(image_folder[i], f'{ID}-{date}', 'image_cropped.npz')
                ori_ID = ID
            else:
                index = ns_image_folder_iddates_list[i].index(f'{ID}-{date}')
                file_path = os.path.join(image_folder[i], image_folder_iddates_list[i][index], 'image_cropped.npz')
                ori_ID = image_folder_iddates_list[i][index].split('-')[0]
            if os.path.exists(file_path):
                if image_flag:
                    try:
                        image_data = np.load(file_path)['arr_0'][5:15]
                    except Exception as e:
                        print(f"[Error loading file] part{i}-{file_path}: {e}")
                        continue
                    if np.max(np.abs(image_data)) > 100000:
                        print(f'The image data of patient {ID} is invalid')
                    # image_data = np.clip(image_data, -3000, 3000)

                    image_data = np.clip(image_data, -1400, 1400)
                    image_data = (image_data + 1400.0) / 2800.0

                covariates_row = [0 if row[gender_col_name] == '男' or row[gender_col_name] == 0 else 1, row[age_col_name], row[height_col_name], row[weight_col_name]]

                if i != 7:
                    responses_row = [row[FEV1_col_name], row[FVC_col_name], row[TLC_col_name]]
                else:

                    FEV1_value = row[FEV1_col_name]
                    FVC_value = row[FVC_col_name]


                    responses_row = [FEV1_value, FVC_value, 0]

                if any(pd.isnull(covariates_row)) or any(pd.isnull(responses_row)):
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} has missing response or covariate (basic).\n')
                    continue
                if responses_row[0] > 10 or responses_row[1] > 10 or responses_row[2] > 10:
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} has invalid response.\n')
                    continue

                if i == 0 or i == 7:
                    report_file_path = os.path.join(report_path[i], f'{ID}', 'ctreport_embedding.npz')
                else:
                    report_file_path = os.path.join(report_path[i], image_folder_iddates_list[i][index], 'ctreport_embedding.npz')
                if not os.path.exists(report_file_path):
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} has no CT report embedding npy file.\n')
                    continue
                if report_flag:
                    report_embedding = np.load(report_file_path)['arr_0']

                if i == 0 or i == 7:
                    reasoning_file_path = os.path.join(LLM_reasoning_path[i], f'{ID}', 'ct_LLM_reasoning_embedding.npz')
                    reasoning_mask_path = os.path.join(LLM_reasoning_path[i], f'{ID}', 'ct_LLM_token_mask.npz')
                else:
                    reasoning_file_path = os.path.join(LLM_reasoning_path[i], image_folder_iddates_list[i][index], 'ct_LLM_reasoning_embedding.npz')
                    reasoning_mask_path = os.path.join(LLM_reasoning_path[i], image_folder_iddates_list[i][index], 'ct_LLM_token_mask.npz')
                if not os.path.exists(reasoning_file_path) or not os.path.exists(reasoning_mask_path):
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} has no LLM reasoning embedding npy file.\n')
                    continue
                if LLM_reasoning:
                    reasoning_embedding = np.load(reasoning_file_path)['arr_0']
                    reasoning_mask = np.load(reasoning_mask_path)['arr_0']

                LLM_df_now = LLM_df_list[i]
                LLM_row = LLM_df_now[LLM_df_now[id_col_name] == ori_ID]
                if LLM_row.empty:
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} is not included in LLM reasoning xlsx.\n')
                    continue
                LLM_row = LLM_row.iloc[0]
                if ((LLM_row == 'Error') | (LLM_row == 'E') | (pd.isna(LLM_row['FEV1大模型答案置信度'])) | (
                        pd.isna(LLM_row['FVC大模型答案置信度'])) | (pd.isna(LLM_row['TLC大模型答案置信度']))).any():
                    with open(log_file_path, "a") as file:
                        file.write(f'Part{i}: Patient: {ID} does not have complete data row in LLM xlsx.\n')
                    continue
                if LLM_predict:
                    covariates_row = covariates_row + [0 if LLM_row['FVC大模型答案'] == '否' else 1, LLM_row['FVC大模型答案置信度'],
                                                       0 if LLM_row['FEV1大模型答案'] == '否' else 1, LLM_row['FEV1大模型答案置信度'],
                                                       0 if LLM_row['TLC大模型答案'] == '否' else 1, LLM_row['TLC大模型答案置信度']]

                if mp_flag:
                    responses_mp_row = []
                    pft_diagnosis_row = []

                    FEV1FVC = (float(responses_row[0]) / float(responses_row[1])) * 100.0
                    responses_mp_row.append(FEV1FVC)
                    pft_diagnosis_row.append(int(FEV1FVC < 70.0))
                    if int(FEV1FVC < 70.0):
                        FEV1FVC_pos_num += 1

                    sex = covariates_row[0]
                    age = covariates_row[1]
                    height = covariates_row[2]
                    # 20260216 update
                    if i != 7:
                        FEV1_pre = PFT_data_predictive_value(sex, height, age, PFT_metric='FEV1')
                        FVC_pre = PFT_data_predictive_value(sex, height, age, PFT_metric='FVC')
                    else:
                        # FEV1_pre = float(row[FEV1_pre_col_name])
                        # FVC_pre = float(row[FVC_pre_col_name])
                        FEV1_pre = PFT_data_predictive_value(sex, height, age, PFT_metric='FEV1')
                        FVC_pre = PFT_data_predictive_value(sex, height, age, PFT_metric='FVC')
                    TLC_pre = PFT_data_predictive_value(sex, height, age, PFT_metric='TLC')

                    FEV1_mp_value = (float(responses_row[0]) / FEV1_pre) * 100.0
                    responses_mp_row.append(FEV1_mp_value)
                    pft_diagnosis_row.append(int(FEV1_mp_value < 80.0))
                    if int(FEV1_mp_value < 80.0):
                        FEV1_pos_num += 1
                    if i == 7:
                        responses_mp_row.append(0)
                        FVC_mp_value = (float(responses_row[1]) / FVC_pre) * 100.0
                        TLC_label = int(FEV1FVC >= 70.0 and FVC_mp_value < 80.0)

                        pft_diagnosis_row.append(TLC_label)
                        if int(TLC_label):
                            TLC_pos_num += 1
                    else:
                        TLC_mp_value = (float(row[TLC_col_name]) / TLC_pre) * 100.0
                        responses_mp_row.append(TLC_mp_value)
                        pft_diagnosis_row.append(int(TLC_mp_value < 80.0))
                        if int(TLC_mp_value < 80.0):
                            TLC_pos_num += 1

                covariates_row = [float(re.sub(r'[^0-9.]', '', str(c))) for c in covariates_row]
                responses_row = [float(r) for r in responses_row]

                if test_set_name1 in ID and len(ID) > 4:
                    if image_flag:
                        images_test_1.append(image_data)
                    ids_test_1.append(ID)
                    dates_test_1.append(date)
                    covariates_test_1.append(covariates_row)
                    responses_test_1.append(responses_row)
                    if report_flag:
                        reports_test_1.append(report_embedding)
                    if LLM_reasoning:
                        reasoning_test_1.append(reasoning_embedding)
                        reasoning_mask_test_1.append(reasoning_mask)
                    if mp_flag:
                        responses_mp_test_1.append(responses_mp_row)
                        diagnosises_test_1.append(pft_diagnosis_row)
                elif test_set_name2 in ID and len(ID) > 4:
                    if image_flag:
                        images_test_2.append(image_data)
                    ids_test_2.append(ID)
                    dates_test_2.append(date)
                    covariates_test_2.append(covariates_row)
                    responses_test_2.append(responses_row)
                    if report_flag:
                        reports_test_2.append(report_embedding)
                    if LLM_reasoning:
                        reasoning_test_2.append(reasoning_embedding)
                        reasoning_mask_test_2.append(reasoning_mask)
                    if mp_flag:
                        responses_mp_test_2.append(responses_mp_row)
                        diagnosises_test_2.append(pft_diagnosis_row)
                else:
                    if ID in IDs_train:
                        if image_flag:
                            images_train.append(image_data)
                        ids_train.append(ID)
                        dates_train.append(date)
                        covariates_train.append(covariates_row)
                        responses_train.append(responses_row)
                        if report_flag:
                            reports_train.append(report_embedding)

                        if LLM_reasoning:
                            reasoning_train.append(reasoning_embedding)
                            reasoning_mask_train.append(reasoning_mask)
                        if mp_flag:
                            responses_mp_train.append(responses_mp_row)
                            diagnosises_train.append(pft_diagnosis_row)
                    elif ID in IDs_val:
                        if image_flag:
                            images_val.append(image_data)
                        ids_val.append(ID)
                        dates_val.append(date)
                        covariates_val.append(covariates_row)
                        responses_val.append(responses_row)
                        if report_flag:
                            reports_val.append(report_embedding)
                        if LLM_reasoning:
                            reasoning_val.append(reasoning_embedding)
                            reasoning_mask_val.append(reasoning_mask)
                        if mp_flag:
                            responses_mp_val.append(responses_mp_row)
                            diagnosises_val.append(pft_diagnosis_row)
                    else:
                        continue

            else:
                continue

    train_data = {
        "ID_list": ids_train,
        "date_list": dates_train,
        "covariates": np.array(covariates_train, dtype=np.float32),
        "responses": np.array(responses_train, dtype=np.float32),
        "responses_mp": np.array(responses_mp_train, dtype=np.float32),
        "diagnosis": np.array(diagnosises_train, dtype=np.float32)
    }

    val_data = {
        "ID_list": ids_val,
        "date_list": dates_val,
        "covariates": np.array(covariates_val, dtype=np.float32),
        "responses": np.array(responses_val, dtype=np.float32),
        "responses_mp": np.array(responses_mp_val, dtype=np.float32),
        "diagnosis": np.array(diagnosises_val, dtype=np.float32)
    }

    test_data_1 = {
        "ID_list": ids_test_1,
        "date_list": dates_test_1,
        "covariates": np.array(covariates_test_1, dtype=np.float32),
        "responses": np.array(responses_test_1, dtype=np.float32),
        "responses_mp": np.array(responses_mp_test_1, dtype=np.float32),
        "diagnosis": np.array(diagnosises_test_1, dtype=np.float32)
    }

    test_data_2 = {
        "ID_list": ids_test_2,
        "date_list": dates_test_2,
        "covariates": np.array(covariates_test_2, dtype=np.float32),
        "responses": np.array(responses_test_2, dtype=np.float32),
        "responses_mp": np.array(responses_mp_test_2, dtype=np.float32),
        "diagnosis": np.array(diagnosises_test_2, dtype=np.float32)
    }
    if image_flag:
        train_data["images"] = np.array(images_train, dtype=np.float32)
        val_data["images"] = np.array(images_val, dtype=np.float32)
        test_data_1["images"] = np.array(images_test_1, dtype=np.float32)
        test_data_2["images"] = np.array(images_test_2, dtype=np.float32)
    else:
        train_data["images"] = None
        val_data["images"] = None
        test_data_1["images"] = None
        test_data_2["images"] = None
    if report_flag:
        train_data["reports"] = np.array(reports_train, dtype=np.float32)
        val_data["reports"] = np.array(reports_val, dtype=np.float32)
        test_data_1["reports"] = np.array(reports_test_1, dtype=np.float32)
        test_data_2["reports"] = np.array(reports_test_2, dtype=np.float32)
    else:
        train_data["reports"] = None
        val_data["reports"] = None
        test_data_1["reports"] = None
        test_data_2["reports"] = None
    if LLM_reasoning:
        train_data["reasonings"] = np.array(reasoning_train, dtype=np.float32)
        val_data["reasonings"] = np.array(reasoning_val, dtype=np.float32)
        test_data_1["reasonings"] = np.array(reasoning_test_1, dtype=np.float32)
        test_data_2["reasonings"] = np.array(reasoning_test_2, dtype=np.float32)

        train_data["reasonings_mask"] = np.array(reasoning_mask_train, dtype=np.float32)
        val_data["reasonings_mask"] = np.array(reasoning_mask_val, dtype=np.float32)
        test_data_1["reasonings_mask"] = np.array(reasoning_mask_test_1, dtype=np.float32)
        test_data_2["reasonings_mask"] = np.array(reasoning_mask_test_2, dtype=np.float32)
    else:
        train_data["reasonings"] = None
        val_data["reasonings"] = None
        test_data_1["reasonings"] = None
        test_data_2["reasonings"] = None

        train_data["reasonings_mask"] = None
        val_data["reasonings_mask"] = None
        test_data_1["reasonings_mask"] = None
        test_data_2["reasonings_mask"] = None

    total_sample_num = train_data["covariates"].shape[0] + val_data["covariates"].shape[0] + test_data_1["covariates"].shape[0] + test_data_2["covariates"].shape[0]
    print("total_sample_num:", total_sample_num)

    print("total FEV1FVC_pos_num:", FEV1FVC_pos_num)
    print("total FEV1_pos_num:", FEV1_pos_num)
    print("total TLC_pos_num:", TLC_pos_num)

    print("total FEV1FVC_pos_rate:", float(FEV1FVC_pos_num) / float(total_sample_num))
    print("total FEV1_pos_rate:", float(FEV1_pos_num) / float(total_sample_num))
    print("total TLC_pos_rate:", float(TLC_pos_num) / float(total_sample_num))

    print("train_FEV1FVC_pos_num:", train_data["diagnosis"].sum(axis=0, keepdims=True)[0][0])
    print("train_FEV1_pos_num:", train_data["diagnosis"].sum(axis=0, keepdims=True)[0][1])
    print("train_TLC_pos_num:", train_data["diagnosis"].sum(axis=0, keepdims=True)[0][2])

    print("val_FEV1FVC_pos_num:", val_data["diagnosis"].sum(axis=0, keepdims=True)[0][0])
    print("val_FEV1_pos_num:", val_data["diagnosis"].sum(axis=0, keepdims=True)[0][1])
    print("val_TLC_pos_num:", val_data["diagnosis"].sum(axis=0, keepdims=True)[0][2])

    print(f"test_data_{test_set_name1}_FEV1FVC_pos_num:", test_data_1["diagnosis"].sum(axis=0, keepdims=True)[0][0])
    print(f"test_data_{test_set_name1}_FEV1_pos_num:", test_data_1["diagnosis"].sum(axis=0, keepdims=True)[0][1])
    print(f"test_data_{test_set_name1}_TLC_pos_num:", test_data_1["diagnosis"].sum(axis=0, keepdims=True)[0][2])

    print(f"test_data_{test_set_name2}_FEV1FVC_pos_num:", test_data_2["diagnosis"].sum(axis=0, keepdims=True)[0][0])
    print(f"test_data_{test_set_name2}_FEV1_pos_num:", test_data_2["diagnosis"].sum(axis=0, keepdims=True)[0][1])
    print(f"test_data_{test_set_name2}_TLC_pos_num:", test_data_2["diagnosis"].sum(axis=0, keepdims=True)[0][2])
    if image_flag:
        print("Train Data Image Shapes:")
        print(train_data["images"].shape)
        print("\nValidation Data Image Shapes:")
        print(val_data["images"].shape)
        print(f"\n{test_set_name1} Test Data Image Shapes:")
        print(test_data_1["images"].shape)
        print(f"\n{test_set_name2} Test Data Image Shapes:")
        print(test_data_2["images"].shape)

    if report_flag:
        print("Train Data Report Shapes:")
        print(train_data["reports"].shape)
        print("\nValidation Data Report Shapes:")
        print(val_data["reports"].shape)
        print(f"\n{test_set_name1} Test Data Report Shapes:")
        print(test_data_1["reports"].shape)
        print(f"\n{test_set_name2} Test Data Report Shapes:")
        print(test_data_2["reports"].shape)

    scaler_responses = StandardScaler()
    scaler_responses.fit(
        np.concatenate([train_data["responses"], val_data["responses"], test_data_1["responses"], test_data_2["responses"]], axis=0))
    train_data["responses_norm"] = scaler_responses.transform(train_data["responses"])
    val_data["responses_norm"] = scaler_responses.transform(val_data["responses"])
    test_data_1["responses_norm"] = scaler_responses.transform(test_data_1["responses"])
    test_data_2["responses_norm"] = scaler_responses.transform(test_data_2["responses"])
    train_data["responses_mean"] = scaler_responses.mean_
    val_data["responses_mean"] = scaler_responses.mean_
    test_data_1["responses_mean"] = scaler_responses.mean_
    test_data_2["responses_mean"] = scaler_responses.mean_
    train_data["responses_scale"] = scaler_responses.scale_
    val_data["responses_scale"] = scaler_responses.scale_
    test_data_1["responses_scale"] = scaler_responses.scale_
    test_data_2["responses_scale"] = scaler_responses.scale_

    return train_data, val_data, test_data_1, test_data_2, [float(FEV1FVC_pos_num) / float(total_sample_num), float(FEV1_pos_num) / float(total_sample_num), float(TLC_pos_num) / float(total_sample_num), float(DLCO_pos_num) / float(total_sample_num)]


def train_val_dataloader(train_data, val_data, test_data_1, test_data_2, random_state=1, batch_size=16, train_data_augment=False, train_mix_data_augment=False, train_dataset_shuffle=True):

    train_dataset = CustomDataset(train_data["covariates"], train_data["responses_norm"], train_data["diagnosis"], images=train_data["images"], ct_reports=train_data["reports"],
                                  reasonings=train_data["reasonings"], reasonings_mask=train_data["reasonings_mask"])
    val_dataset = CustomDataset(val_data["covariates"], val_data["responses_norm"], val_data["diagnosis"],
                                  images=val_data["images"], ct_reports=val_data["reports"],
                                  reasonings=val_data["reasonings"], reasonings_mask=val_data["reasonings_mask"])
    test_dataset_1 = CustomDataset(test_data_1["covariates"], test_data_1["responses_norm"], test_data_1["diagnosis"],
                                images=test_data_1["images"], ct_reports=test_data_1["reports"],
                                reasonings=test_data_1["reasonings"], reasonings_mask=test_data_1["reasonings_mask"])
    test_dataset_2 = CustomDataset(test_data_2["covariates"], test_data_2["responses_norm"], test_data_2["diagnosis"],
                                images=test_data_2["images"], ct_reports=test_data_2["reports"],
                                reasonings=test_data_2["reasonings"], reasonings_mask=test_data_2["reasonings_mask"])

    if train_data_augment:
        if train_mix_data_augment:
            aug_train_dataset_ro_zo = CustomDataset(train_data["covariates"], train_data["responses_norm"],
                                                  train_data["diagnosis"], images=train_data["images"],
                                                  ct_reports=train_data["reports"],
                                                  reasonings=train_data["reasonings"],
                                                  reasonings_mask=train_data["reasonings_mask"],
                                                  image_augment=True, angle=10, scale_range=0.05)
        else:
            aug_train_dataset_ro = CustomDataset(train_data["covariates"], train_data["responses_norm"],
                                                    train_data["diagnosis"], images=train_data["images"],
                                                    ct_reports=train_data["reports"],
                                                    reasonings=train_data["reasonings"],
                                                    reasonings_mask=train_data["reasonings_mask"],
                                                    image_augment=True, angle=10)
            aug_train_dataset_zo = CustomDataset(train_data["covariates"], train_data["responses_norm"],
                                                    train_data["diagnosis"], images=train_data["images"],
                                                    ct_reports=train_data["reports"],
                                                    reasonings=train_data["reasonings"],
                                                    reasonings_mask=train_data["reasonings_mask"],
                                                    image_augment=True, scale_range=0.05)

    if train_data_augment:
        if train_mix_data_augment:
            train_dataset = aug_train_dataset_ro_zo
        else:
            train_dataset = ConcatDataset([train_dataset, aug_train_dataset_ro, aug_train_dataset_zo])

    g1 = torch.Generator().manual_seed(random_state)
    g2 = torch.Generator().manual_seed(random_state)
    g3 = torch.Generator().manual_seed(random_state)
    g4 = torch.Generator().manual_seed(random_state)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_dataset_shuffle, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=random_state), generator=g1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=random_state), generator=g2)
    test_loader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=random_state), generator=g3)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=random_state), generator=g4)

    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, 1 test set : {len(test_dataset_1)}, 2 test set : {len(test_dataset_2)}")

    return train_loader, val_loader, test_loader_1, test_loader_2

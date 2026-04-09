import pandas as pd
from transformers import BertTokenizer, BertModel
import re
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import jieba
from collections import Counter
import pkuseg
from gensim import corpora, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import json

if __name__ == '__main__':
    ctr_df_path = '/teams/Thymoma_1685081756/PFT/data/PFT (M+P) -Cov-CTreport-Class-DSR1-MedP.xlsx'
    ctr_df3_path = '/teams/Thymoma_1685081756/PFT/data/PFT_summary_20250322_1254.xlsx'
    ctr_df4_path = '/teams/Thymoma_1685081756/PFT/data/PFT_summary_20250423_1959.xlsx'
    ctr_df5_path = '/course75/RealData/PFT_summary_20250520_898.xlsx'

    embedding_save_path = '/teams/Thymoma_1685081756/PFT/data/ct_report_20CharMean_embedding'
    embedding3_save_path = '/teams/Thymoma_1685081756/PFT/data/ct_report_20CharMean_embedding_part3'
    embedding4_save_path = '/teams/Thymoma_1685081756/PFT/data/ct_report_20CharMean_embedding_part4'
    embedding5_save_path = '/teams/Thymoma_1685081756/PFT/data/ct_report_20CharMean_embedding_part5'

    ctr_df = pd.read_excel(ctr_df_path).copy()
    ctr_df3 = pd.read_excel(ctr_df3_path).copy()
    ctr_df4 = pd.read_excel(ctr_df4_path).copy()
    ctr_df5 = pd.read_excel(ctr_df5_path).copy()

    seg = pkuseg.pkuseg(model_name='/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/model_ckpts/pkuseg/medicine')
    df_list = [ctr_df, ctr_df3, ctr_df4, ctr_df5]
    see_diagnosis_words_list = []
    word_dict = []
    word_l2_dict = []
    word_pao_dict = []
    covs_list = []
    diags_list = []
    for i, df in enumerate(df_list):
        if i == 0:
            id_col_name = 'new_ID'
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
        else:
            id_col_name = 'Patient_ID'
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
        for j, row in df.iterrows():
            id = str(row[id_col_name])
            if 'CJJ' not in id and 'CT' not in id and 'P' not in id:
                id = id.zfill(10)
            ct_saw_diagnosis = [row[des_col_name], row[diag_col_name]]
            covs = [0 if row[gender_col_name] == '男' or row[gender_col_name] == 0 else 1, row[age_col_name], row[weight_col_name], row[height_col_name]]
            merics = [row[FEV1_col_name], row[FEV1_pre_col_name], row[FVC_col_name], row[TLC_col_name], row[TLC_pre_col_name]]
            if any(pd.isna(ct_saw_diagnosis)) or any(pd.isna(covs)) or any(pd.isna(merics)):
                continue
            if i == 0:
                FEV1_diag = 1 if float(merics[1]) / 100.0 < 0.8 else 0
                FVC_diag = 1 if float(merics[0]) / float(merics[2]) < 0.7 else 0
                TLC_diag = 1 if float(merics[4]) / 100.0 < 0.8 else 0
            else:
                FEV1_diag = 1 if float(merics[0]) / float(merics[1]) < 0.8 else 0
                FVC_diag = 1 if float(merics[0]) / float(merics[2]) < 0.7 else 0
                TLC_diag = 1 if float(merics[3]) / float(merics[4]) < 0.8 else 0
            covs = [float(cov) for cov in covs]
            diags = [FEV1_diag, FVC_diag, TLC_diag]

            covs_list.append(covs)
            diags_list.append(diags)

            see_diagnosis = ''.join(ct_saw_diagnosis)
            see_diagnosis = re.sub(r'[A-Za-z0-9]', '', see_diagnosis)
            sentences = re.split(r'[。，；、;,:]', see_diagnosis)
            sentences = [re.sub(r'[- ×=*\.\?？≈/㎝\\]', '', s) for s in sentences if s.strip()]
            # for ch in ['-', ' ', '=', '*', '.', '×']:
            #     see_diagnosis = see_diagnosis.replace(ch, '')
            # see_diagnosis = re.sub(r'[^\u4e00-\u9fa5]', '', see_diagnosis)
            # tokens = jieba.lcut(see_diagnosis)
            tokens = []
            for s in sentences:
                tokens += seg.cut(s)

            #tokens = seg.cut(see_diagnosis)
            # tokens = graces.cut(see_diagnosis)
            tokens_gl3 = [token for token in tokens if len(token) > 2]
            tokens_pao = [token for token in tokens if '泡' in token]
            see_diagnosis_words_list.append(tokens_gl3)
            word_dict += tokens_gl3
            word_pao_dict += tokens_pao
    covs_M = np.array(covs_list)
    diags_M = np.array(diags_list)

    word_black_list = ['结节', '磨玻璃', '玻璃', '脉', '血管', '软组织', '肺门', '叶',
                       '甲状腺', '乳', '肠', '胃', '胆', '骨', '脊', '椎', '肾', '心', '肝', '淋巴', '腔', '尿', '消化道', '食管', "胰腺",
                       '区', '期', '术', '步', '形', '值', '灶', '类', '状', '型', '实性', '段', '面', '毛刺', '点',
                       '基底', '规则', '起搏器', '部分', '密度']
    word_black_strict_list = ["支气管", "纵隔窗", "可能性", "对称性", "局限性", "细支气管", "气管壁", "左下肺", "右下肺", "支气管壁",
                              "余气管", "混合磨", "实验室", "低剂量", "左主支气管", "右主支气管", "左肺支气管", "右肺支气管", "左下肺肺", "小支气管"]

    uni_word_dict = list(set(word_dict))
    uni_word_pao_dict = list(set(word_pao_dict))
    filtered_word_dict = []
    filtered_see_diagnosis_words_list = []
    for see_diagnosis_words in see_diagnosis_words_list:
        filtered_see_diagnosis_words = [word for word in see_diagnosis_words
                                        if word in uni_word_dict and
                                        all(word_b not in word for word_b in word_black_list) and
                                        all(word_b != word for word_b in word_black_strict_list)]
        filtered_see_diagnosis_words_list.append(filtered_see_diagnosis_words)
        filtered_word_dict += filtered_see_diagnosis_words
    filtered_word_dict = list(set(filtered_word_dict))

    # filtered_see_diagnosis_words_list = see_diagnosis_words_list
    # filtered_word_dict = list(set(word_dict))
    word_freq = Counter(filtered_word_dict)
    top_words = word_freq.most_common()

    w_num = len(filtered_word_dict)
    AUC_M = np.zeros((w_num + 1, diags_M.shape[1]))
    X = covs_M
    Y = diags_M
    word_M = np.zeros((X.shape[0], w_num))
    for i, filtered_word in enumerate(filtered_word_dict):
        for j, filtered_see_diagnosis_words in enumerate(filtered_see_diagnosis_words_list):
            if filtered_word in filtered_see_diagnosis_words:
                word_M[j, i] = 1.0
            else:
                word_M[j, i] = 0.0
    X_train, X_test, word_M_train, word_M_test, Y_train, Y_test = train_test_split(X, word_M, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    for i in range(Y.shape[1]):
        Y_train_now = Y_train[:, i]
        Y_test_now = Y_test[:, i]
        model.fit(X_train, Y_train_now)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test_now, y_prob)
        AUC_M[0, i] = auc

    for i in range(w_num):
        X_train_now = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
        X_train_now[:, :X_train.shape[1]] = X_train
        X_train_now[:, X_train.shape[1]] = word_M_train[:, i]

        X_test_now = np.zeros((X_test.shape[0], X_test.shape[1] + 1))
        X_test_now[:, :X_test.shape[1]] = X_test
        X_test_now[:, X_test.shape[1]] = word_M_test[:, i]
        for j in range(Y.shape[1]):
            Y_train_now = Y_train[:, j]
            Y_test_now = Y_test[:, j]

            model.fit(X_train_now, Y_train_now)
            y_prob = model.predict_proba(X_test_now)[:, 1]
            auc = roc_auc_score(Y_test_now, y_prob)
            AUC_M[i+1, j] = auc

    AUC_inc_M = AUC_M - AUC_M[0, :]
    AUC_inc_mean = np.mean(AUC_inc_M, axis=1)[1:]
    AUC_inc_fev1 = AUC_inc_M[1:, 0]
    AUC_inc_fvc = AUC_inc_M[1:, 1]
    AUC_inc_tlc = AUC_inc_M[1:, 2]

    mean_sort_indces = np.argsort(-AUC_inc_mean)
    fev1_sort_indces = np.argsort(-AUC_inc_fev1)
    fvc_sort_indces = np.argsort(-AUC_inc_fvc)
    tlc_sort_indces = np.argsort(-AUC_inc_tlc)

    all_auc_sorted_dicts = {
        "mean_auc_sorted_dict": {
            filtered_word_dict[i]: AUC_inc_mean[i]
            for i in mean_sort_indces
        },
        "fev1_auc_sorted_dict": {
            filtered_word_dict[i]: AUC_inc_fev1[i]
            for i in fev1_sort_indces
        },
        "fvc_auc_sorted_dict": {
            filtered_word_dict[i]: AUC_inc_fvc[i]
            for i in fvc_sort_indces
        },
        "tlc_auc_sorted_dict": {
            filtered_word_dict[i]: AUC_inc_tlc[i]
            for i in tlc_sort_indces
        }
    }

    with open("./result_images/word_lg3_black_screen/auc_sorted_dicts.json", "w", encoding="utf-8") as f:
        json.dump(all_auc_sorted_dicts, f, ensure_ascii=False, indent=4)

    mean_auc_top_dict = {k: v for k, v in all_auc_sorted_dicts["mean_auc_sorted_dict"].items() if v > 0}
    fev1_auc_top_dict = {k: v for k, v in all_auc_sorted_dicts["fev1_auc_sorted_dict"].items() if v > 0}
    fvc_auc_top_dict = {k: v for k, v in all_auc_sorted_dicts["fvc_auc_sorted_dict"].items() if v > 0}
    tlc_auc_top_dict = {k: v for k, v in all_auc_sorted_dicts["tlc_auc_sorted_dict"].items() if v > 0}

    dict_list = [mean_auc_top_dict, fev1_auc_top_dict, fvc_auc_top_dict, tlc_auc_top_dict]
    wcimg_name_list = ['mean_auc_word_cloud.png', 'fev1_auc_word_cloud.png', 'fvc_auc_word_cloud.png', 'tlc_auc_word_cloud.png']
    for i, dict in enumerate(dict_list):
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ).generate_from_frequencies(dict)

        wordcloud.to_file('./result_images/word_lg3_black_screen/' + wcimg_name_list[i])

        max_word = max(dict, key=dict.get)
        min_word = min(dict, key=dict.get)
        print(wcimg_name_list[i].split('.')[0])
        print(f"max incremental AUC: {max_word}-{dict[max_word]}")
        print(f"min incremental AUC: {min_word}-{dict[min_word]}")

    print('Job Done')


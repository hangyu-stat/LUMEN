import pandas as pd
from transformers import BertTokenizer, BertModel
import re
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import jieba
import pkuseg
import json

if __name__ == '__main__':
    ctr_df_path = ''
    ctr_df3_path = ''
    ctr_df4_path = ''
    ctr_df5_path = ''
    ctr_df6_path = ''
    ctr_df7_path = ''
    ctr_df8_path = ''
    ctr_df9_path = ''

    embedding_save_path = ''
    embedding3_save_path = ''
    embedding4_save_path = ''
    embedding5_save_path = ''
    embedding6_save_path = ''
    embedding7_save_path = ''
    embedding8_save_path = ''
    # 20260216 update
    embedding9_save_path = ''

    embedding_save_path_list = [embedding_save_path, embedding3_save_path, embedding4_save_path, embedding5_save_path, embedding6_save_path, embedding7_save_path, embedding8_save_path, embedding9_save_path]
    word_auc_json_path = './result_images/word_lg3_black_screen/auc_sorted_dicts.json'
    with open(word_auc_json_path, 'r', encoding='utf-8') as f:
        all_auc_dicts = json.load(f)
    seg = pkuseg.pkuseg(model_name='./models/pkuseg/medicine')

    ctr_df = pd.read_excel(ctr_df_path).copy()
    ctr_df3 = pd.read_excel(ctr_df3_path).copy()
    ctr_df4 = pd.read_excel(ctr_df4_path).copy()
    ctr_df5 = pd.read_excel(ctr_df5_path).copy()
    ctr_df6 = pd.read_excel(ctr_df6_path).copy()
    ctr_df7 = pd.read_excel(ctr_df7_path).copy()
    ctr_df8 = pd.read_excel(ctr_df8_path).copy()
    # 20260216 update
    ctr_df9 = pd.read_excel(ctr_df9_path).copy()
    ctr_df_list = [ctr_df, ctr_df3, ctr_df4, ctr_df5, ctr_df6, ctr_df7, ctr_df8, ctr_df9]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('./models/medbert-base-chinese')
    model = BertModel.from_pretrained('./models/medbert-base-chinese')
    model.to(device)

    embedding_list = []
    label_list = []
    null_text_num = 0
    ct_report_change_num = 0
    for i, df in enumerate(ctr_df_list):
        # 20260216 update
        if i not in [7]:
            continue
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
        # 20260216 update
        elif i == 7:
            id_col_name = 'Patient_ID'
            date_col_name = ''
            gender_col_name = 'gender'
            age_col_name = 'age'
            weight_col_name = 'weight'
            height_col_name = 'height'
            des_col_name = 'Description'
            diag_col_name = 'Diagnosis'
            FEV1_col_name = 'FEV1_实测值'
            FEV1_pre_col_name = ''
            FVC_col_name = 'FVC_实测值'
            FVC_pre_col_name = ''
            TLC_col_name = ''
            TLC_pre_col_name = ''
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
            CT_report_change_name = 'CTreport_change'
        for j, row in df.iterrows():
            id = str(row[id_col_name])
            '''
            LLM_row = LLM_df[LLM_df[id_col_name] == id]
            if LLM_row.empty:
                continue
            else:
                LLM_row = LLM_row.iloc[0]
                FEV1_reason = LLM_row['FEV1大模型推理']
                # print('FEV1大模型推理:\n', FEV1_reason)
            '''

            if i != 0:
                # 20260216 update
                if not 'P' in id and not 'CT' in id and not 'CJJ' in id and not 'YX' in id:
                    id = id.zfill(10)
            if i == 0 or i == 7:
                date = ''
            else:
                date = str(row[date_col_name])
                if i == 5:
                    date = date.split(' ')[0]
                    date = "".join(date.split('-'))
            # detect ct report change
            '''
            ct_change = int(row[CT_report_change_name])
            if not ct_change:
                continue
            else:
                print(f'{id} patient has a ct report change.')
                ct_report_change_num += 1
            '''

            ct_saw_diagnosis = [row[des_col_name], row[diag_col_name]]
            # 20260216 update
            # metrics = [row[FEV1_col_name], row[FVC_col_name]]

            if all(pd.isnull(ct_saw_diagnosis)):
                print(f'{id} patient has no report.')
                continue
            # FEV1FVC = 1 if float(metrics[0]) / float(metrics[1]) < 0.7 else 0
            see_diagnosis = ''.join([x if pd.notnull(x) else '' for x in ct_saw_diagnosis])
            # print("see_diagnosis:\n", see_diagnosis)
            see_diagnosis = re.sub(r'[A-Za-z0-9]', '', see_diagnosis)
            sentences = re.split(r'[。，；、;,:\?？]', see_diagnosis)
            sentences = [re.sub(r'[- ×=*\.≈/㎝\\]', '', s) for s in sentences if s.strip()]
            # tokens = jieba.lcut(see_diagnosis)
            tokens = []
            for s in sentences:
                tokens += seg.cut(s)
            filtered_tokens = []
            for token in tokens:
                '''
                if token in all_auc_dicts["mean_auc_sorted_dict"] and all_auc_dicts["mean_auc_sorted_dict"][token] > 0:
                    filtered_tokens.append(token)
                    continue
                '''
                if token in all_auc_dicts["fev1_auc_sorted_dict"] and all_auc_dicts["fev1_auc_sorted_dict"][token] > 0:
                    filtered_tokens.append(token)
                    continue
                if token in all_auc_dicts["fvc_auc_sorted_dict"] and all_auc_dicts["fvc_auc_sorted_dict"][token] > 0:
                    filtered_tokens.append(token)
                    continue
                if token in all_auc_dicts["tlc_auc_sorted_dict"] and all_auc_dicts["tlc_auc_sorted_dict"][token] > 0:
                    filtered_tokens.append(token)
                    continue
            # print("filtered_tokens:\n", filtered_tokens)
            if len(filtered_tokens) >= 1:
                if len(filtered_tokens) == 1:
                    filtered_see_diagnosis = filtered_tokens[0]
                else:
                    filtered_see_diagnosis = ' '.join(list(set(filtered_tokens)))
                have_text = 1
                inputs = tokenizer(filtered_see_diagnosis, return_tensors='pt', padding=True, truncation=True,
                                   max_length=48).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                see_diagnosis_embedding = outputs.last_hidden_state[:, 0, :]
                see_diagnosis_embedding_npy = see_diagnosis_embedding.cpu().numpy()
                see_diagnosis_embedding_npy = np.concatenate([np.array([[have_text]]), see_diagnosis_embedding_npy], axis=1)
            else:
                filtered_see_diagnosis = ''
                null_text_num += 1
                have_text = 0
                see_diagnosis_embedding_npy = np.zeros((1, 769))
                # print(f'Patient {id} report has no text after filtering.')

            embedding_list.append(see_diagnosis_embedding_npy)  # shape: (1, 769)
            # label_list.append(FEV1FVC)  # 1 for <0.7 (red), 0 for >=0.7 (blue)

            if i == 0 or i == 7:
                save_dir = os.path.join(embedding_save_path_list[i], f"{id}")
            else:
                save_dir = os.path.join(embedding_save_path_list[i], f"{id}-{date}")

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savez_compressed(os.path.join(save_dir, 'ctreport_embedding.npz'), see_diagnosis_embedding_npy)






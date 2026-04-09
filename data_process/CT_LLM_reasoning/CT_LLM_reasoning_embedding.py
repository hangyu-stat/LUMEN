import pandas as pd
from transformers import BertTokenizer, BertModel
import re
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ctrllmr_df_path = ''
    ctrllmr_df3_path = ''
    ctrllmr_df4_path = ''
    ctrllmr_df5_path = ''
    ctrllmr_df6_path = ''
    ctrllmr_df7_path = ''
    ctrllmr_df8_path = ''
    ctrllmr_df9_path = ''

    embedding_save_path = ''
    embedding3_save_path = ''
    embedding4_save_path = ''
    embedding5_save_path = ''
    embedding6_save_path = ''
    embedding7_save_path = ''
    embedding8_save_path = ''
    embedding9_save_path = ''

    ctr_df = pd.read_csv(ctrllmr_df_path, encoding='utf-8').copy()
    ctr_df3 = pd.read_csv(ctrllmr_df3_path, encoding='utf-8').copy()
    ctr_df4 = pd.read_csv(ctrllmr_df4_path, encoding='utf-8').copy()
    ctr_df5 = pd.read_csv(ctrllmr_df5_path, encoding='utf-8').copy()
    ctr_df6 = pd.read_csv(ctrllmr_df6_path, encoding='utf-8').copy()
    ctr_df7 = pd.read_csv(ctrllmr_df7_path, encoding='utf-8').copy()
    ctr_df8 = pd.read_csv(ctrllmr_df8_path, encoding='utf-8').copy()
    # 20260216 update
    ctr_df9 = pd.read_csv(ctrllmr_df9_path, encoding='utf-8').copy()
    ctr_df_list = [ctr_df, ctr_df3, ctr_df4, ctr_df5, ctr_df6, ctr_df7, ctr_df8, ctr_df9]
    embedding_save_path_list = [embedding_save_path, embedding3_save_path, embedding4_save_path, embedding5_save_path, embedding6_save_path, embedding7_save_path, embedding8_save_path, embedding9_save_path]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    tokenizer = BertTokenizer.from_pretrained('./models/medbert-base-chinese')
    model = BertModel.from_pretrained('./models/medbert-base-chinese')
    model.to(device)

    embedding_list = []
    label_list = []

    for i, df in enumerate(ctr_df_list):
        # remember to change when data updates

        if i != 7:
            continue

        if i == 0:
            id_col_name = 'new_ID'
            date_col_name = ''
            gender_col_name = '性别'
            age_col_name = '年龄'
            weigth_col_name = '体重'
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
            weigth_col_name = 'weight'
            height_col_name = 'height'
            des_col_name = 'CT所见'
            diag_col_name = 'CT诊断'
            FEV1_col_name = 'FEV 1_实测值'
            FEV1_pre_col_name = 'FEV 1_预计值'
            FVC_col_name = 'FVC_实测值'
            FVC_pre_col_name = 'FVC_预计值'
            TLC_col_name = 'TLC_实测值'
            TLC_pre_col_name = 'TLC_预计值'
        for j, row in df.iterrows():
            id = str(row[id_col_name])
            if i != 0:
                # 20260216 update
                if not 'P' in id and not 'CT' in id and not 'CJJ' in id and not 'YD' in id and not 'YX' in id:
                    id = id.zfill(10)
            if i == 0 or i == 7:
                date = ''
            else:
                date = str(row[date_col_name])
                if i == 5:
                    date = date.split(' ')[0]
                    date = "".join(date.split('-'))

            reasoning_FEV1 = row['FEV1大模型关键推理总结']
            reasoning_FVC = row['FVC大模型关键推理总结']
            reasoning_TLC = row['TLC大模型关键推理总结']

            reasonings = [reasoning_FEV1, reasoning_FVC, reasoning_TLC]
            if any(pd.isnull(reasonings)) or 'Error' in reasonings:
                continue

            if len(reasonings) == 0:
                continue
 
            max_length = 128
            inputs = tokenizer(reasonings, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                outputs = model(**inputs)
            # sentences_embedding = outputs.last_hidden_state.mean(dim=1)
            sentences_embedding = outputs.last_hidden_state
            std_sentences_embedding_npy = sentences_embedding.cpu().numpy()
            # embedding_list.append(std_sentences_embedding_npy[0, :])

            if i == 0 or i == 7:
                save_dir = os.path.join(embedding_save_path_list[i], f"{id}")
            else:
                save_dir = os.path.join(embedding_save_path_list[i], f"{id}-{date}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savez_compressed(os.path.join(save_dir, 'ct_LLM_reasoning_embedding.npz'), std_sentences_embedding_npy)
            np.savez_compressed(os.path.join(save_dir, 'ct_LLM_token_mask.npz'), attention_mask.cpu().numpy())





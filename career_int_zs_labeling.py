# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 10:07:02 2022

@author: akatz4
"""





# import transformers


# from transformers import pipeline


# from personal_utilities import embed_cluster as ec
from personal_utilities import zs_labeling as zsl

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


import pickle


"""

Import data and preprocess

"""

topic = "career_interests"
# run_date = "20221210"
# run_date = "20221212"
# run_date = "20221213"
# run_date = "20221214"
run_date = "20221215"

os.getcwd()

# for jee df with abstracts and full article information
proj_path = 'G:\My Drive\AK Faculty\Research\Projects\project student career interests\Results\label testing'

os.chdir(proj_path)
os.listdir()




text_sentence_df = pd.read_csv("career_interests_aug_sampled_count_5rts_0.8thres_n7_big_ex_mpnet_20221126.csv")
print(text_sentence_df.columns)

text_sentence_df['new_sent_id'] = text_sentence_df.index
unlabeled_df = text_sentence_df
unlabeled_df




# =============================================================================
# Load labels
# =============================================================================


# labels_df = pd.read_csv("career_interests_summary_augmented.csv")
labels_df = pd.read_csv("career_interests_summary_augmented - labels.csv")

labels_df.columns


class_labels = list(labels_df.label_v3.dropna().unique())
print(len(class_labels))
class_labels




"""

Labeling

"""


filtered_df = unlabeled_df.dropna(subset=['split_sent'])


test_df = filtered_df.sample(n=200, random_state=42)



# =============================================================================
# using the utility file zs_labeling.py and label_df_with_zs()
# =============================================================================


zs_threshold = 0.1

text_col_name = 'split_sent'
id_col_name = 'new_sent_id'
multi_label = False
keep_top_n = True
top_n = 5

total_results_df = zsl.label_df_with_zs(test_df, 
                                        text_col_name, 
                                        id_col_name, 
                                        class_labels, 
                                        zs_threshold, 
                                        multi_label=multi_label,
                                        keep_top_n=keep_top_n,
                                        top_n=top_n)




zs_threshold_save = str(zs_threshold).replace('.', '-')

if multi_label == True:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_multi_top{top_n}_{run_date}.csv", index = False)

if multi_label == False:
    total_results_df.to_csv(f"{topic}_zs_label_{zs_threshold_save}thresh_no-multi_{run_date}.csv", index = False)

# os.chdir("C:\\Users\\akatz4\\OneDrive - Virginia Tech\\Desktop")
# os.listdir()
# total_results_df.to_csv("_zs_label_.csv", index = False)







"""

Sandbox

"""

classifier = pipeline(task = 'zero-shot-classification', model = 'facebook/bart-large-mnli')

test_text = "By being a roller coaster design engineer, I would be working on fixing many problems."

class_labels.append('Roller coasters')
class_labels

classifier_results = classifier(test_text, class_labels)
classifier_results




# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:03:42 2019

@author: MAGESHWARAN
"""
import os
import json
import numpy as np


base_dir = os.getcwd()
data_folder = os.path.join(base_dir, "Dataset")
sample_testset = os.path.join(data_folder, "sample_testset")

model_sample_result = os.path.join(sample_testset, "sample_result.json")
gt_sample_result = os.path.join(sample_testset, "out_res.json")

with open(model_sample_result, "r") as f:
    model_result = json.load(f)

with open(gt_sample_result, "r") as f:
    gt_result = json.load(f)

gt_images = list(gt_result.keys())
model_images = list(model_result.keys())

tp = tn = fp = fn = 0

for img in gt_images:

    gt_association = gt_result[img]
    model_association = model_result[img]
    gt_list = []
    model_list = []

    if len(gt_association) > 0:
        for i in range(len(gt_association)):
            gt_list.append(gt_association[i][0])

        for j in range(len(model_association)):
            model_list.append(model_association[j][0])

        gt_copy = gt_list.copy()
        model_copy = model_list.copy()

        for association in gt_list:
            if association in model_list:
                if img != "NA":
                    tp += 1
                else:
                    tn += 1
                gt_copy.remove(association)
                model_copy.remove(association)

            else:
                fn += 1

        for found in model_copy:
            if found not in gt_copy:
                fp += 1

    elif len(model_association) == 0:
        tn += 1

    elif len(model_association) > 0:
        fp += len(model_association)

precision = tp / (tp + fp)

recall = tp / (tp + fn)

f1_score = 2* ((precision * recall) / (precision + recall))

print("Precision: ", precision)
print("recall: ", recall)
print("F1 Score:", f1_score)

confusion_matrix = np.array(([tp, fp], [fn, tn]))

print("Confusion Matrix:", confusion_matrix)
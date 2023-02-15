#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:40:19 2020
Modified pn Mon Feb 08 13:21:00 2021

@author: neerajs/srikanthr
"""

import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def score(reference_labels, sys_scores, thresholds=np.arange(0, 1, 0.0001)):
    # Arrays to store true positives, false positives, true negatives, false negatives
    tp = np.zeros((len(reference_labels), len(thresholds)))
    tn = np.zeros((len(reference_labels), len(thresholds)))
    if type(sys_scores) == dict:
        key_cnt = -1
        for key in sys_scores:  # Repeat for each recording
            key_cnt += 1
            sys_labels = (sys_scores[key] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
            gt = reference_labels[key]

            ind = np.where(sys_labels == gt)  # system label matches the ground truth
            if gt == 1:  # ground-truth label=1: True positives
                tp[key_cnt, ind] = 1
            else:  # ground-truth label=0: True negatives
                tn[key_cnt, ind] = 1

        total_positives = sum(reference_labels.values())  # Total number of positive samples
        total_negatives = len(reference_labels) - total_positives  # Total number of negative samples
    elif type(sys_scores) == list:
        for key_cnt in range(len(sys_scores)):  # Repeat for each recording
            sys_labels = (sys_scores[key_cnt] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
            gt = reference_labels[key_cnt]

            ind = np.where(sys_labels == gt)  # system label matches the ground truth
            if gt == 1:  # ground-truth label=1: True positives
                tp[key_cnt, ind] = 1
            else:  # ground-truth label=0: True negatives
                tn[key_cnt, ind] = 1

        total_positives = sum(reference_labels)  # Total number of positive samples
        total_negatives = len(reference_labels) - total_positives  # Total number of negative samples
    else:
        raise ValueError('unknown input type, expecting a list or dict type')

    tp = np.sum(tp, axis=0)  # Sum across the recordings
    tn = np.sum(tn, axis=0)

    tpr = tp / total_positives  # True positive rate: #true_positives/#total_positives
    tnr = tn / total_negatives  # True negative rate: #true_negatives/#total_negatives

    auc_score = auc(1 - tnr, tpr)  # auc_score
    acc = total_positives / (total_positives + total_negatives)

    return acc, auc_score, tpr, tnr


def scoring(refs, sys_outs, out_file=None, specificities_chosen=None):
    """
    inputs::
    refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
    sys_outs: a txt file with a list of scores (probability of being covid positive)
     for each wav-fileid in the format: <id> <score>
    out_file (optional): name of the output file
    specificities_chosen: optionally mention the specificities at which sensitivity is reported

    """
    # Read data and get labels and scores
    y_score, y_true, reference_labels, sys_scores = get_labels_and_scores(refs, sys_outs)
    # Calculate the auc_score, FP-rate, and TP-rate
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, n_thresholds = roc_curve(y_true, y_score)
    # calculate the specificity and sensitivity
    specificity = tpr[0]
    sensitivity = tpr[1]
    print(f'Specificity: {specificity:.3f}')
    print(f'Sensitivity: {sensitivity:.3f}')

    # Make prediction using the better threshold
    y_pred = [1 if scr > n_thresholds[-1] else 0 for scr in y_score]
    # Calculate Precision, Recall, and F-beta scores
    precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred)
    # Calculate F1 score
    f1_scr = f1_score(y_true, y_pred)
    # Calculate Confusion Matrix
    confusion_mx = confusion_matrix(y_true, y_pred)

    # %%
    if specificities_chosen is None:
        specificities_chosen = [0.55, 0.95]

    thresholds = np.arange(0, 1, 0.0001)

    acc_score, auc_score, tpr, tnr = score(reference_labels, sys_scores, thresholds=thresholds)

    specificities = []
    sensitivities = []
    decision_thresholds = []
    for specificity_threshold in specificities_chosen:
        ind = np.where(tnr > specificity_threshold)[0]
        sensitivities.append(tpr[ind[0]])
        specificities.append(tnr[ind[0]])
        decision_thresholds.append(thresholds[ind[0]])

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except auc_score) is a array for different threshold values
    # Specificity at 95% sensitivity
    scores = {'UC': auc_score,
              'TPR': tpr,
              'FPR': 1 - tnr,
              'sensitivity': sensitivities,
              'specificity': specificities,
              'operatingPts': decision_thresholds,
              'thresholds': thresholds,
              'ACC': acc_score,
              'F1': f1_score}

    if out_file is not None:
        # Save scores
        with open(out_file, "wb") as f:
            pickle.dump(scores, f)
        # Save scores as a human-readable txt file
        with open(out_file.replace('.pkl', '.summary'), 'w') as f:
            pretty_score = f"acc_score {acc_score:.3f}\t" \
                           f"auc_score {auc_score:.3f}\t " \
                           f"Sens({1 - specificities_chosen[0]:.2f}). {sensitivities[0]:.3f}\t" \
                           f"Spec({specificities_chosen[0]:.2f}). {specificities[0]:.3f}\t" \
                           f"Sens({1 - specificities_chosen[1]:.2f}). {sensitivities[1]:.3f}\t" \
                           f"Spec({specificities_chosen[1]:.2f}). {specificities[1]:.3f}\n"
            f.write(pretty_score)
            print(pretty_score)

        # Plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # Save the ROC curve plot
        plt.savefig(out_file.replace('.pkl', '_ROC.png'))

        # Plot the confusion matrix for the threshold selected
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.savefig(out_file.replace('.pkl', '_confusion_matrix.png'))
        plt.close()

    return scores


def get_labels_and_scores(refs, sys_outs):
    # Read the ground truth labels into a dictionary
    reference_labels = {}
    categories = ['n', 'p']
    data = open(refs).readlines()
    for line in data:
        key, val = line.strip().split()
        reference_labels[key] = categories.index(val)
    # Read the system scores into a dictionary
    sys_scores = {}
    data = open(sys_outs).readlines()
    for line in data:
        key, val = line.strip().split()
        sys_scores[key] = float(val)
    del data
    # Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        raise ValueError(
            "Expected the score file to have scores for all files in reference and no duplicates/extra entries")
    # %%
    y_true, y_score = [], []
    for key in reference_labels.keys():
        y_true.append(reference_labels[key])
        y_score.append(sys_scores[key])
    return y_score, y_true, reference_labels, sys_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_file', '-r', default='data/reference')
    parser.add_argument('--target_file', '-t', default='results/wav_test_scores.txt')
    parser.add_argument('--output_file', '-o', default='results/wav_test_scores.pkl')
    args = parser.parse_args()

    scoring(args.ref_file, args.target_file, args.output_file)

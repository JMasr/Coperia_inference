#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:40:19 2020
Modified pn Mon Feb 08 13:21:00 2021

@author: neerajs/srikanthr

Edited on 2022
@author: José Manuel Ramírez (@github/JMasr)

This script is used to calculate the performance metrics for the covid-19 and long-covid detection task.
"""

import pickle
import argparse
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix, \
    precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay


def read_label_and_score_files(label_file: str, score_file: str) -> dict:
    """
    Read the label and score files, and return a dictionary with the id as key and the label and score as values
    :param label_file: file with the labels <id> <label>
    :param score_file:  file with the scrs <id> <score>
    :return: a dictionary with the id as key and the label and score as values {<id>: [<label>,<score>] }
    """
    # Read the ground truth labels into a dictionary
    sys_out = {}
    categories = ['n', 'p']
    labels = sorted(open(label_file).readlines())
    scrs = sorted(open(score_file).readlines())

    for label, scr in zip(labels, scrs):
        id_label, label = label.strip().split()
        id_score, scr = scr.strip().split()

        if id_label != id_score:
            raise ValueError("Expected the label file and score file to have the same ids")
        sys_out[id_label] = [categories.index(label), float(scr)]

    del labels, scrs
    return sys_out


def filter_labels(system_output: dict, labels: list) -> dict:
    """
    Filter the dictionary to include only the labels of interest
    :param system_output: dictionary of labels and scores
    :param labels: list with the labels of interest
    :return: a dictionary of labels and scores
    """
    return {k: v for k, v in system_output.items() if k in labels}


def score_sklearn(out_file, system_output):
    """
    Calculate the performance metrics using sklearn
    :param out_file: Path to the output files
    :param system_output: dictionary of labels and scores
    :return: a set of performance metrics: confusion matrix, f1 score, f-beta score, precision, recall, and auc score
    """
    # Get the labels and scores
    y_score = [scr for _, scr in system_output.values()]
    y_true = [label for label, _ in system_output.values()]

    # Calculate the auc_score, FP-rate, and TP-rate
    sklearn_roc_auc_score = roc_auc_score(y_true, y_score)
    sklear_fpr, sklearn_tpr, n_thresholds = roc_curve(y_true, y_score)

    # calculate the specificity and sensitivity
    sensitivity = sklearn_tpr[1]
    specificity = 1 - sensitivity
    print(f'Specificity: {specificity:.3f}')
    print(f'Sensitivity: {sensitivity:.3f}')

    # Make prediction using the better threshold
    y_pred = [1 if scr > n_thresholds[-1] else 0 for scr in y_score]

    # Calculate Precision, Recall, F1, and F-beta scores
    precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred)
    f1_scr = f1_score(y_true, y_pred)

    # Calculate Confusion Matrix
    confusion_mx = confusion_matrix(y_true, y_pred)

    # Plot useful metric graphs
    if out_file is not None:
        # Plot the ROC curve for the model
        plt.plot(sklear_fpr, sklearn_tpr, marker='.', label='Logistic')
        plt.title('ROC Curve')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # Save the ROC curve plot
        plt.savefig(out_file.replace('.pkl', '_ROC.png'))
        plt.close()

        # Plot the Precision-Recall curve for the model
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(precision, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the title
        plt.title('Precision-Recall Curve')
        plt.savefig(out_file.replace('.pkl', '_precision_recall.png'))
        plt.close()

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mx, display_labels=['Negative', 'Positive'])
        disp.plot()
        # Plot the Confusion Matrix for the threshold selected
        plt.title('Confusion Matrix')
        plt.savefig(out_file.replace('.pkl', '_confusion_matrix.png'))
        plt.close()

    return confusion_mx, f1_scr, f_beta, precision, recall, sklearn_roc_auc_score


def score(system_output: dict, thresholds: np.arange = np.arange(0, 1, 0.0001)):
    """
    Calculate the performance metrics for a range of thresholds
    :param system_output: dictionary with the id as key and the label and score as values {<id>: [<label>,<score>] }
    :param thresholds: a range of thresholds
    :return: a set of performance metrics acc_score, auc_score, tpr, and tnr
    """
    # Arrays to store true positives, false positives, true negatives, false negatives
    tp = np.zeros((len(system_output), len(thresholds)))
    tn = np.zeros((len(system_output), len(thresholds)))

    key_cnt = -1
    for key in system_output.keys():  # Repeat for each recording
        key_cnt += 1
        sys_labels = (system_output[key][1] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
        gt = system_output[key][0]  # Ground truth label

        ind = np.where(sys_labels == gt)  # system label matches the ground truth
        if gt == 1:  # ground-truth label=1: True positives
            tp[key_cnt, ind] = 1
        else:  # ground-truth label=0: True negatives
            tn[key_cnt, ind] = 1

    total_positives = sum([label for label, _ in system_output.values()])  # Total number of positive samples
    total_negatives = len(system_output) - total_positives  # Total number of negative samples

    tp = np.sum(tp, axis=0)  # Sum across the recordings
    tn = np.sum(tn, axis=0)  # Sum across the recordings

    tpr = tp / total_positives  # True positive rate: #true_positives/#total_positives
    tnr = tn / total_negatives  # True negative rate: #true_negatives/#total_negatives

    acc = total_positives / (total_positives + total_negatives)  # Accuracy: #total_positives/#total_samples

    return acc, tpr, tnr


def scoring(refs: str, sys_outs: str, out_file: str = None, spc_chosen: list = None, thr: np.arange = None) -> dict:
    """
    inputs::
    refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
    sys_outs: a txt file with a list of dict_scores (probability of being covid positive)
     for each wav-fileid in the format: <id> <score>
    out_file (optional): name of the output file
    specificities_chosen: optionally mention the specificities at which sensitivity is reported

    """
    # Default values for thresholds
    if thr is None:
        thr = np.arange(0, 1, 0.0001)

    # Default values for specificities
    if spc_chosen is None:
        spc_chosen = [0.55, 0.95]

    # Read data and get labels and dict_scores
    system_output = read_label_and_score_files(refs, sys_outs)

    # Calculate the performance metrics using sklearn
    confusion_mx, f1_scr, f_beta, precision, recall, auc_score = score_sklearn(out_file, system_output)

    # Calculate the performance metrics for a range of thresholds
    acc_score, tpr, tnr = score(system_output, thresholds=thr)

    # Calculate the performance metrics for specificities
    specificities, sensitivities, decision_thresholds = [], [], []
    for specificity_threshold in spc_chosen:
        ind = np.where(tnr > specificity_threshold)[0]
        sensitivities.append(tpr[ind[0]])
        specificities.append(tnr[ind[0]])
        decision_thresholds.append(thr[ind[0]])

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except auc_score) is a array for different threshold values
    # Specificity at 95% sensitivity
    dict_scores = {'acc_score': acc_score,
                   'tpr': tpr,
                   'tnr': tnr,
                   'sensitivities': sensitivities,
                   'specificities': specificities,
                   'decision_thresholds': decision_thresholds,
                   'auc_score': auc_score,
                   'confusion_mx': confusion_mx,
                   'f1_scr': f1_scr,
                   'f_beta': f_beta,
                   'precision': precision,
                   'recall': recall}

    if out_file is not None:
        # Save dict_scores
        with open(out_file, "wb") as f:
            pickle.dump(dict_scores, f)
        # Save dict_scores as a human-readable txt file
        with open(out_file.replace('.pkl', '.summary'), 'w') as f:
            pretty_score = f"ACC {acc_score:.3f}\t" \
                           f"AUC {auc_score:.3f}\t " \
                           f"Sens({1 - spc_chosen[0]:.2f}). {sensitivities[0]:.3f}\t" \
                           f"Spec({spc_chosen[0]:.2f}). {specificities[0]:.3f}\t" \
                           f"Sens({1 - spc_chosen[1]:.2f}). {sensitivities[1]:.3f}\t" \
                           f"Spec({spc_chosen[1]:.2f}). {specificities[1]:.3f}\n"
            f.write(pretty_score)
            print(pretty_score)

    return dict_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    reference_labels = '/home/jmramirez/Documentos/COPERIA/codes-in-tool-release/classifier_BLSTM_model/data/val'
    scores = '/home/jmramirez/Documentos/COPERIA/models/coswara_model_by_tasks/results_lr/cough-heavy/val_scores.txt'
    parser.add_argument('--ref_file', '-r', default=reference_labels)
    parser.add_argument('--target_file', '-t', default=scores)
    parser.add_argument('--output_file', '-o', default='results/wav_test_scores.pkl')
    args = parser.parse_args()

    scoring(args.ref_file, args.target_file, args.output_file)

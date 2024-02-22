import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import xlrd
from sklearn.metrics import confusion_matrix

from sklearn import metrics
import csv
def get_column_elements(file_path, column_index_output, column_index_answer):

    with open(file_path,encoding='utf-8',errors='ignore') as csvfile:

        # Return a reader object which will
        # iterate over lines in the given csvfile.
        readCSV = csv.reader(csvfile, delimiter=',')
        # for row in readCSV:
        #     print(row)
        #     print(row[0])
        #     print(row[0], row[1], row[2], )
        column_elements_output = []
        column_elements_answer = []
        for row_index in readCSV:  # 从第二行开始读取，避免读取表头
            if row_index[column_index_output] == "true_label":
                continue
            elif row_index[column_index_answer] == "pre_label":
                continue
            column_elements_output.append(int(row_index[column_index_output])-1)
            column_elements_answer.append(int(row_index[column_index_answer])-1)
        return column_elements_output,column_elements_answer

def eval_performance(y_true, y_pred):
    # Precision
    print("Precision:\n\t", metrics.precision_score(y_true, y_pred, average='weighted'))

    # Recall
    print("Recall:\n\t", metrics.recall_score(y_true, y_pred, average='weighted'))

    # Accuracy
    print("Accuracy:\n\t", metrics.accuracy_score(y_true, y_pred))

    print("----------F1, Micro-F1, Macro-F1, Weighted-F1..----------------")
    print("----------**********************************----------------")

    # F1 Score
    print("F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    # Micro-F1 Score
    print("Micro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='micro'))

    # Macro-F1 Score
    print("Macro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='macro'))

    # Weighted-F1 Score
    print("Weighted-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    print("------------------**********************************-------------------------")
    print("-------------------**********************************-------------------------")

    # ROC AUC Score
    # print("ROC AUC:\n\t", metrics.roc_auc_score(y_true, y_pred))

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))


file1_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test19.csv"
file2_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test29.csv"
file3_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test39.csv"
file4_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test49.csv"
file5_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test59.csv"
file6_path = r"C:\Users\ASIA\Desktop\BoostingLLM\mid_result\STT2_test69.csv"
y_true_index = 1
y_pred_index = 2
y_true, pred_1 = get_column_elements(file1_path, y_true_index,  y_pred_index)
y_true, pred_2 = get_column_elements(file2_path, y_true_index,  y_pred_index)
y_true, pred_3 = get_column_elements(file3_path, y_true_index,  y_pred_index)
y_true, pred_4 = get_column_elements(file4_path, y_true_index,  y_pred_index)
y_true, pred_5 = get_column_elements(file5_path, y_true_index,  y_pred_index)
y_true, pred_6 = get_column_elements(file6_path, y_true_index,  y_pred_index)
pred_lebel=[]
i=0
w=[]

for j in y_true:
    predict = w[0]*pred_1[i] + w[1]*pred_2[i]+w[2]*pred_3[i]+w[3]*pred_4[i]+w[4]*pred_5[i]+w[5]*pred_6[i]
    print(predict)
    if predict > 0.5:
        pred = 1
    else:
        pred = 0
    pred_lebel.append(pred)
    i=i+1;
eval_performance(y_true, pred_lebel)
print(y_true)
print(pred_1)

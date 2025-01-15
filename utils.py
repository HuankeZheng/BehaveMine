import numpy as np


def cal_accuracy(result, len_activity):
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    TP = np.zeros(len_activity)
    FP = np.zeros(len_activity)
    FN = np.zeros(len_activity)
    P = np.zeros(len_activity)
    R = np.zeros(len_activity)
    F1 = np.zeros(len_activity)
    num_type = np.zeros(len_activity)
    for item in result:
        num_type[item['act_target']] += 1
        if item['act_target'] == item['act_pred']:
            correct_top1 += 1
            TP[item['act_target']] += 1
        else:
            FP[item['act_pred']] += 1
            FN[item['act_target']] += 1
        if item['act_target'] in item['top5_indices'][-3:]:
            correct_top3 += 1
        if item['act_target'] in item['top5_indices']:
            correct_top5 += 1

    accuracy_top1 = correct_top1 / len(result)
    accuracy_top3 = correct_top3 / len(result)
    accuracy_top5 = correct_top5 / len(result)

    Macro_F1 = 0
    for i in range(len_activity):
        if num_type[i] == 0 or TP[i] == 0:
            continue
        P[i] = TP[i] / (TP[i] + FP[i])
        R[i] = TP[i] / (TP[i] + FN[i])
        F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])
        Macro_F1 += F1[i]

    Macro_F1 = Macro_F1 / len_activity
    accuracy = {'top1': accuracy_top1, 'top3': accuracy_top3, 'top5': accuracy_top5}
    return accuracy, Macro_F1


def split_data(data, train_ratio=0.6, test_ratio=0.2):
    train_data = data[:int(len(data) * train_ratio)]
    test_data = data[int(len(data) * train_ratio):int(len(data) * (train_ratio + test_ratio))]
    eval_data = data[int(len(data) * (train_ratio + test_ratio)):]
    return train_data, test_data, eval_data

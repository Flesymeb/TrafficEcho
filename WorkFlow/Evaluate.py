import json
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss, jaccard_score

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    true_labels = [item['true_label'] for item in data]
    predicted_labels = [item['predicted_label'] for item in data]
    return true_labels, predicted_labels

# 转换为二进制矩阵
def labels_to_binary_matrix(labels_list, all_labels):
    label_to_index = {label: i for i, label in enumerate(all_labels)}
    binary_matrix = np.zeros((len(labels_list), len(all_labels)), dtype=int)
    for i, labels in enumerate(labels_list):
        for label in labels:
            if label in label_to_index:
                binary_matrix[i, label_to_index[label]] = 1
    return binary_matrix

if __name__ == "__main__":
    file_path = "NEclassified_data_noreason.json"
    true_labels, predicted_labels = load_data(file_path)
    all_labels = sorted(set(label for labels in true_labels + predicted_labels for label in labels))
    y_true = labels_to_binary_matrix(true_labels, all_labels)
    y_pred = labels_to_binary_matrix(predicted_labels, all_labels)
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    report = classification_report(y_true, y_pred, target_names=all_labels, zero_division=0)
    print(f"Accuracy:{accuracy:.4f}")
    print(f"Hamming Loss:{hamming:.4f}")
    print(f"Jaccard Similarity:{jaccard:.4f}")
    print(f"Macro F1-Score:{macro_f1:.4f}")
    print(f"Micro F1-Score:{micro_f1:.4f}")
    print("\nClassification Report:\n")
    print(report)
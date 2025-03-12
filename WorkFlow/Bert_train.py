import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
from tqdm import tqdm

# 读取数据
with open('classified_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取文本和标签
texts = [item["微博正文"] for item in data]
labels = [item["true_label"] for item in data]

# 处理多标签：转换为 one-hot 编码
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(labels)  # 将标签转换为 one-hot

# 训练集/测试集划分
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, label_matrix, test_size=0.4, random_state=42
)

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


# 数据集类
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }


# 生成 DataLoader
batch_size = 8
train_dataset = WeiboDataset(train_texts, train_labels, tokenizer)
test_dataset = WeiboDataset(test_texts, test_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 加载 BERT 预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=len(mlb.classes_), problem_type="multi_label_classification"
).to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

loss_fn = torch.nn.BCEWithLogitsLoss()  # 多标签任务使用 BCEWithLogitsLoss

# 早停策略
class EarlyStopping:
    def __init__(self, patience=3, min_acc=0.99):
        self.patience = patience
        self.min_acc = min_acc
        self.counter = 0
        self.best_acc = 0

    def should_stop(self, acc):
        if acc >= self.min_acc:  # 训练集准确率 >= 90%
            print(f"训练准确率达到 {acc:.4f}, 触发早停!")
            return True
        return False

# 初始化早停
early_stopping = EarlyStopping()

# 训练循环
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    # 使用 tqdm 进度条
    train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in train_dataloader_tqdm:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算训练准确率
        preds = torch.sigmoid(outputs.logits).cpu().detach().numpy()
        preds = (preds > 0.5).astype(int)  # 设定阈值
        labels_np = labels.cpu().numpy()

        batch_acc = accuracy_score(labels_np, preds)  # 批量准确率
        correct += batch_acc * len(labels)
        total += len(labels)

        train_dataloader_tqdm.set_postfix(loss=loss.item(), acc=batch_acc)

    # 计算整个训练集的准确率
    train_acc = correct / total
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 早停检查
    if early_stopping.should_stop(train_acc):
        break  # 触发早停

# 评估模型
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (preds > 0.5).astype(int)  # 设定阈值，将概率转换为二进制类别

        all_preds.extend(preds)
        all_labels.extend(labels)

# 转换为 NumPy 数组
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 计算分类报告
print("\n**分类报告 (Classification Report)**:")
print(classification_report(all_labels, all_preds, target_names=mlb.classes_))

# 计算汉明损失
hamming = hamming_loss(all_labels, all_preds)
print(f"\n**汉明损失 (Hamming Loss)**: {hamming:.4f}")

# 计算 Jaccard 相似性
jaccard = jaccard_score(all_labels, all_preds, average='samples')  # 逐样本计算 Jaccard 相似度
print(f"\n**Jaccard 相似性 (Jaccard Similarity Score)**: {jaccard:.4f}")

# 计算 Exact Match Accuracy（严格准确率）
exact_match_acc = np.all(all_preds == all_labels, axis=1).mean()
print(f"\n**严格准确率 (Exact Match Accuracy)**: {exact_match_acc:.4f}")



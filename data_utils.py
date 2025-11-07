import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text
        }
    def gettexts(self):
        return self.texts
def load_and_split_dataset(dataset_name, tokenizer, max_samples, max_length, num_clients):
    """Load dataset, split it for federated learning, and create client datasets"""
    if dataset_name == 'imdb':
        texts, labels = load_imdb_dataset(max_samples)
        num_labels = 2
    elif dataset_name == 'agnews':
        texts, labels = load_agnews_dataset(max_samples)
        num_labels = 4
    elif dataset_name == 'bbcnews':
        texts, labels = load_bbcnews_dataset(max_samples)
        num_labels = 5
    elif dataset_name == 'reuters':
        texts, labels = load_reuters_dataset(max_samples)
        num_labels = 8
    # Medical datasets
    elif dataset_name == 'pubmed_rct':
        texts, labels = load_pubmed_rct_dataset(max_samples)
        num_labels = 5
    elif dataset_name == 'medical_abstracts':
        texts, labels = load_medical_abstracts_dataset(max_samples)
        num_labels = 5
    elif dataset_name == 'mnli_resampled_as_mednli':
        load_mnli_resampled_as_mednli_dataset
        texts, labels = load_mnli_resampled_as_mednli_dataset(max_samples)
        num_labels = 3 
    # Legal datasets
    elif dataset_name == 'scotus':
        texts, labels = load_scotus_dataset(max_samples)
        num_labels = 13
    elif dataset_name == 'ecthr':
        texts, labels = load_ecthr_dataset(max_samples)
        num_labels = 8
    elif dataset_name == 'mimic_attitude':
        texts, labels = load_mimic_attitude_dataset(max_samples)
        num_labels = 3  # 正式设置为3类
    elif dataset_name == 'medical_questions_pairs':
        texts, labels = load_medical_questions_pairs(max_samples, tokenizer=tokenizer)
        num_labels = 2  
    elif dataset_name == 'ade_corpus':
        texts, labels = load_ade_corpus(max_samples)
        num_labels = 2  
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.4, stratify=labels, random_state=42
    )

    # Create test dataset
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)

    # Split training data for clients (non-iid split with Dirichlet distribution)
    client_data = split_data_for_clients(X_train, y_train, num_clients, num_labels)

    # Create client datasets
    client_datasets = []
    for i in range(num_clients):
        client_X, client_y = client_data[i]
        client_dataset = TextDataset(client_X, client_y, tokenizer, max_length)
        client_datasets.append(client_dataset)

    return client_datasets, test_dataset

def load_imdb_dataset(max_samples=None, max_len = None):
    """Load IMDB dataset with optional sample limit"""
    dataset = load_dataset("imdb")
    texts = []
    labels = []
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Ensure class balance
        pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
        neg_indices = [i for i, label in enumerate(train_labels) if label == 0]
        
        samples_per_class = max_samples // 2
        selected_pos = pos_indices[:samples_per_class]
        selected_neg = neg_indices[:samples_per_class]
        
        selected_indices = selected_pos + selected_neg
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

def load_agnews_dataset(max_samples=None, max_len = None):
    """Load AG News dataset with optional sample limit"""
    dataset = load_dataset("ag_news")
    texts = []
    labels = []
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(4)]  # AG News has 4 classes
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 4
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

def load_bbcnews_dataset(max_samples=None, max_len = None):
    """Load BBC News dataset with optional sample limit"""
    dataset = load_dataset("SetFit/bbc-news")

    # 首先正确定义基础数据
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    texts, labels = [], []

    # 验证原始数据标签
    assert all(0 <= l < 5 for l in train_labels), "原始数据包含无效标签"

    # 创建类别索引（这才是class_indices的正确定义）
    class_indices = [[] for _ in range(5)]  # 5个类别
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)  # 收集每个类别的索引

    # 限制样本逻辑
    if max_samples and max_samples < len(train_texts):
        # 保证每个类别至少有1个样本
        samples_per_class = max(max_samples // 5, 1)
        
        selected_indices = []
        for class_list in class_indices:
            if len(class_list) > 0:
                selected = class_list[:samples_per_class]
                selected_indices.extend(selected)
        
        # 如果样本不足，随机补充
        if len(selected_indices) < max_samples:
            remaining = max_samples - len(selected_indices)
            extra_indices = random.sample(
                [i for i in range(len(train_texts)) if i not in selected_indices],
                remaining
            )
            selected_indices.extend(extra_indices)
        
        # 最终获取数据
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    # 最终验证
    assert len(texts) == len(labels), "数据与标签长度不一致"
    assert all(0 <= l < 5 for l in labels), f"发现无效标签: {set(l for l in labels if not 0<=l<5)}"
    print("Unique labels:", set(labels))

    return texts, labels

def load_reuters_dataset(max_samples=None, max_len = None):
    """Load Reuters dataset with optional sample limit"""
    dataset = load_dataset("yangwang825/reuters-21578")
    texts = []
    labels = []

    # Get data from training set
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels


    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(8)]  # Reuters has 8 classes
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 8
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

# Medical Dataset 1: PubMed RCT
def load_pubmed_rct_dataset(max_samples=None, max_len=None):
    """
    Load PubMed RCT (Randomized Controlled Trials) dataset
    This dataset classifies sentences from medical abstracts into 5 categories:
    background, objective, method, result, and conclusion
    """
    dataset = load_dataset("pietrolesci/pubmed-200k-rct", "default")
    texts = []
    labels = []

    # Get data from training set
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["labels"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels


    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(5)]  # PubMed RCT has 5 classes
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 5
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

# Medical Dataset 2: Medical Abstracts
def load_medical_abstracts_dataset(max_samples=None, max_len=None):
    """
    Load Medical Abstracts Dataset 
    This dataset contains medical abstracts categorized by medical conditions
    """
    dataset = load_dataset("TimSchopf/medical_abstracts", "default")
    texts = []
    labels = []

    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["medical_abstract"] for example in dataset["train"]]
    original_labels = [example["condition_label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Map string labels to integers
    unique_labels = list(set(train_labels))
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Convert string labels to integers
    train_labels = [label_map[label] for label in train_labels]

    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(5)]  # 5 medical specialties
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 5
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

# Medical Dataset 3: mimic_attitude_dataset
def load_mimic_attitude_dataset(max_samples=None, max_len=None):
    """
    Load Medical Abstracts Dataset 
    这个数据集主要分为三类
    """
    dataset = load_dataset("tanoManzo/mimic_attitude_dataset", "default")
    texts = []
    labels = []

    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Map string labels to integers
    unique_labels = list(set(train_labels))
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Convert string labels to integers
    train_labels = [label_map[label] for label in train_labels]

    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        num_classes=len(label_map)
        class_indices = [[] for _ in range(num_classes)]  # 5 medical specialties
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 5
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels


def load_mnli_resampled_as_mednli_dataset(max_samples=None, max_len=None):
    """
    Load mnli_resampled_as_mednli Dataset
    这是一个三分类的自然语言推理数据集 (entailment, neutral, contradiction)
    """
    dataset = load_dataset("cnut1648/mnli_resampled_as_mednli")

    # 原始数据：前提和假设拼接作为输入文本，label为类别
    original_texts, original_labels = [], []

    # Get data from training set
    train_texts = []
    train_labels = []
    for ex in dataset["train"]:
        original_texts.append(f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}")
        original_labels.append(ex["label"])


    # original_texts = [f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}" 
    #                   for ex in dataset["train"]]
    # original_labels = [ex["label"] for ex in dataset["train"]]

    # --- 长度筛选 ---
    if max_len is not None:
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        train_texts = original_texts
        train_labels = original_labels

    # Map labels (entailment, neutral, contradiction) to integers
    unique_labels = list(set(train_labels))
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    train_labels = [label_map[label] for label in train_labels]

    # --- 限制样本数 ---
    if max_samples and max_samples < len(train_texts):
        num_classes = len(label_map)
        class_indices = [[] for _ in range(num_classes)]
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)

        samples_per_class = max_samples // num_classes
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])

        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

def load_medical_questions_pairs(max_samples=None, tokenizer=None):
    
    if tokenizer.sep_token is None:
        separator = f" {tokenizer.eos_token} "
    else:
        separator = f" {tokenizer.seq_token} "
    dataset = load_dataset("curaihealth/medical_questions_pairs")
    # 2. 从训练集中获取原始数据
    # 这个数据集的输入是两个问题，所以我们分别提取
    train_q1s  = [example["question_1"] for example in dataset["train"]]
    train_q2s  = [example["question_2"] for example in dataset["train"]]
    train_labels  = [example["label"] for example in dataset["train"]]
    if max_samples and max_samples < len(train_labels):
        # Ensure class balance
        pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
        neg_indices = [i for i, label in enumerate(train_labels) if label == 0]
        
        samples_per_class = max_samples // 2
        selected_pos = pos_indices[:samples_per_class]
        selected_neg = neg_indices[:samples_per_class]
        
        selected_indices = selected_pos + selected_neg
        texts = [
            f"{train_q1s[i]} || {train_q2s[i]}"
            for i in selected_indices
        ]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = [
            f"{q1} || {q2}"
            for q1, q2 in zip(train_q1s, train_q2s)
        ]
        labels = train_labels
    '''
    if max_samples and max_samples < len(train_labels):
        num_classes = 2
        class_indices = [[] for _ in range(num_classes)]
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // num_classes
        selected_indices = []
        for class_idx_list in class_indices:
            num_to_take = min(samples_per_class, len(class_idx_list))
            selected_indices.extend(class_idx_list[:num_to_take])
        
        # 将选中的问题对拼接成单一字符串
        texts = [
            f"{train_q1s[i]}{tokenizer.sep_token}{train_q2s[i]}"
            for i in selected_indices
        ]
        labels = [train_labels[i] for i in selected_indices]
    else:
        # 将所有问题对拼接成单一字符串
        texts = [
            f"{q1}{tokenizer.sep_token}{q2}"
            for q1, q2 in zip(train_q1s, train_q2s)
        ]
        labels = train_labels
    '''
    return texts, labels

def load_scotus_dataset(max_samples=None, max_len=None):
    """
    Load SCOTUS (Supreme Court of the United States) dataset
    This dataset contains Supreme Court case opinions categorized by issue area
    """
    dataset = load_dataset("coastalchp/scotus")
    texts = []
    labels = []

    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Check the number of unique labels
    unique_labels = set(train_labels)
    num_classes = len(unique_labels)
    
    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(num_classes)]
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // num_classes
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels
    
    # Print info for debugging
    print(f"SCOTUS dataset loaded: {len(texts)} samples with {num_classes} classes")
    print(f"Label distribution: {sorted([(label, train_labels.count(label)) for label in unique_labels])}")

    return texts, labels



# Legal Dataset 2: European Court of Human Rights (ECtHR)
def load_ecthr_dataset(max_samples=None, max_len=None):
    """
    Load European Court of Human Rights (ECtHR) dataset
    This dataset contains cases with violation or non-violation of articles 
    of the European Convention on Human Rights
    """
    dataset = load_dataset("MHGanainy/ecthr_clustering", "chunks_w_clusters")
    texts = []
    labels = []
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [" ".join(example["text"]) for example in dataset["train"]]  # Join text parts
    original_labels = [example["cluster_id"] for example in dataset["train"]]  # Get the cluster ID as the label

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels
    
    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Find unique labels
        unique_labels = set()
        for label in train_labels:
            if isinstance(label, list):
                # If label is still a list, use the first element
                label_value = label[0] if label else 0
                unique_labels.add(label_value)
            else:
                unique_labels.add(label)
        
        num_classes = len(unique_labels)
        samples_per_class = max_samples // num_classes
        
        # Try to balance classes
        class_indices = {label: [] for label in unique_labels}
        
        for i, label in enumerate(train_labels):
            if isinstance(label, list):
                # If label is still a list, use the first element
                label_value = label[0] if label else 0
            else:
                label_value = label
            
            if label_value in class_indices:
                class_indices[label_value].append(i)
        
        selected_indices = []
        for label_indices in class_indices.values():
            selected_indices.extend(label_indices[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

def load_ade_corpus(max_samples=None, max_len = None):
    """Load IMDB dataset with optional sample limit"""
    dataset = load_dataset("SetFit/ade_corpus_v2_classification")
    texts = []
    labels = []
    train_texts = []
    train_labels = []

    # 从训练集中获取原始数据
    original_texts = [example["text"] for example in dataset["train"]]
    original_labels = [example["label"] for example in dataset["train"]]

    # --- 新增：根据 maxlen 筛选数据 ---
    if max_len is not None:
        # 使用 zip 同时遍历文本和标签，确保它们保持对应关系
        # 我们假设“长度”指的是单词数量，因此使用 text.split()
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(original_texts, original_labels):
            if len(text.split()) < max_len:
                filtered_texts.append(text)
                filtered_labels.append(label)
        
        # 后续操作将基于筛选后的数据进行
        train_texts = filtered_texts
        train_labels = filtered_labels
    else:
        # 如果不限制长度，则使用原始数据
        train_texts = original_texts
        train_labels = original_labels

    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Ensure class balance
        pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
        neg_indices = [i for i, label in enumerate(train_labels) if label == 0]
        
        samples_per_class = max_samples // 2
        selected_pos = pos_indices[:samples_per_class]
        selected_neg = neg_indices[:samples_per_class]
        
        selected_indices = selected_pos + selected_neg
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    return texts, labels

def split_data_for_clients(X, y, num_clients, num_classes, alpha=1):
    """
    Split data for clients using Dirichlet distribution for non-IID setting
    alpha controls the degree of non-IID (lower = more skewed)
    """
    client_data = [[] for _ in range(num_clients)]

    # Group data by class
    class_idxs = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        class_idxs[label].append(idx)

    # For each class, distribute data to clients according to Dirichlet distribution
    for class_idx, idxs in enumerate(class_idxs):
        # Generate distribution for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Calculate number of samples per client for this class
        client_sample_sizes = (np.array(proportions) * len(idxs)).astype(int)
        client_sample_sizes[-1] = len(idxs) - np.sum(client_sample_sizes[:-1])  # Ensure we use all samples
        
        # Distribute indices
        start_idx = 0
        for client_idx in range(num_clients):
            end_idx = start_idx + client_sample_sizes[client_idx]
            client_data[client_idx].extend(idxs[start_idx:end_idx])
            start_idx = end_idx

    # Create final client datasets
    final_client_data = []
    for client_idx in range(num_clients):
        client_indices = client_data[client_idx]
        client_X = [X[i] for i in client_indices]
        client_y = [y[i] for i in client_indices]
        final_client_data.append((client_X, client_y))

    return final_client_data

# Update dataset number of labels dictionary

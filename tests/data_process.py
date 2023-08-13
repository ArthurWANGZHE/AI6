import torch
import numpy as np

def preprocess_text(text, seq_length):
    # 构建字符到数字的映射字典和数字到字符的映射字典
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # 将文本转换为数字序列
    data = [char_to_idx[ch] for ch in text]

    # 构建训练样本
    num_sequences = len(data) - seq_length
    sequences = []
    for i in range(num_sequences):
        seq = data[i:i + seq_length + 1]
        sequences.append(seq)

    return sequences, char_to_idx, idx_to_char



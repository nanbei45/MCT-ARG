import re

import torch
from torch.utils.data import Dataset,DataLoader

# 氨基酸字典
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}  # 从1开始，0作为padding
# 假设结构信息包含alpha螺旋(H)，beta折叠(E)和随机卷曲(C)
structure_to_idx = {'H': 0, 'E': 1, 'C': 2}
def structure_to_indices(structure_seq, structure_to_idx):
    return [structure_to_idx[struct] for struct in structure_seq]
# 假设输入结构信息
structures = ["HHCCE", "CEHHH", "CCCEE"]
# 转换为索引
structures_idx = [structure_to_indices(struct, structure_to_idx) for struct in structures]

def create_attention_mask(padded_sequences):
    """生成attention mask，填充部分为0，序列部分为1"""
    return (padded_sequences != 0).long()

# 氨基酸序列编码函数
def encode_protein_sequence(sequence):
    """将氨基酸序列转换为数值表示"""
    return [AA_TO_INDEX[aa] for aa in sequence]

class ARGsDataset(Dataset):
    # 提取所有的数据
    def __init__(self,sequences,labels):
        self.sequences= sequences
        self.labels = labels

    # 返回所创建数据库的长度
    def __len__(self):
        return len(self.sequences)

    # 创建每一组数据的索引
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return torch.tensor(encode_protein_sequence(sequence)), torch.tensor(label)




import torch
from torch.utils.data import Dataset, DataLoader

# 定义自定义的数据集类
class ARGDataset(Dataset):
    def __init__(self, sequences, labels, max_length):
        self.sequences = self.pad_sequences(self.tokenize_sequences(sequences), max_length)
        self.labels = labels

    # 将氨基酸序列转化为数字序列
    def tokenize_sequences(self, sequences):
        # 假设我们已经定义了一个氨基酸字典
        amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
        tokenizer = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # 0保留给 padding

        def tokenize_sequence(sequence):
            return [tokenizer[aa] for aa in sequence if aa in tokenizer]

        return [tokenize_sequence(seq) for seq in sequences]

    # 填充或截断序列以适应 max_length
    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                padded_sequences.append(seq[:max_length])
            else:
                padded_sequences.append(seq + [0] * (max_length - len(seq)))
        return padded_sequences

    # 数据集长度
    def __len__(self):
        return len(self.sequences)

    # 取出一个样本
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long),torch.tensor(self.labels[idx], dtype=torch.long)


def parse_data(file_path):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                # 去掉开头的 '>'，然后分割字符串
                parts = line[1:].strip().split('|')
                label = int(parts[-1])  # 读取标签
                labels.append(label)
            else:
                seq = line[:-1]
                sequences.append(seq)
    return labels,sequences


class ARGDataset_multi_channel(Dataset):
    def __init__(self, aa_sequences, structure_sequences,rsa_sequences, labels, max_length):
        """
        参数:
        aa_sequences: 氨基酸序列列表
        structure_sequences: 二级结构序列列表
        rsa_sequences: rsa信息
        labels: 标签列表
        max_length: 序列的最大长度
        """
        # 添加有效长度记录
        self.aa_valid_lengths = [len(seq) for seq in aa_sequences]
        self.aa_sequences = self.pad_sequences(self.tokenize_sequences(aa_sequences, 'aa'), max_length)
        self.structure_sequences = self.pad_sequences(self.tokenize_sequences(structure_sequences, 'structure'), max_length)
        self.rsa_sequences = self.pad_sequences(self.normalize_rsa(rsa_sequences), max_length)
        self.labels = labels


    # 将氨基酸和结构序列转化为数字序列
    def tokenize_sequences(self, sequences, mode='aa'):
        """
        mode = 'aa': 处理氨基酸序列
        mode = 'structure': 处理二级结构序列
        """
        if mode == 'aa':
            amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'  # 假设定义了氨基酸字典
            tokenizer = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # 0保留给 padding
        elif mode == 'structure':
            structures = 'HGIEBTSC'  # H=α螺旋, G=310螺旋,I-π-螺旋,E-延伸的 β-链,B-孤立的 β-桥 ,T-转角,S-弯曲,C-无规卷曲
            tokenizer = {struct: idx+1 for idx, struct in enumerate(structures)}  # 0保留给 padding
        else:
            raise ValueError("Invalid mode, use 'aa' or 'structure'.")

        def tokenize_sequence(sequence):
            return [tokenizer[char] for char in sequence if char in tokenizer]

        return [tokenize_sequence(seq) for seq in sequences]

    # 归一化RSA序列
    def normalize_rsa(self, rsa_sequences):
        """
        归一化RSA值到0到1之间。
        """

        # 提取所有值
        all_values = [value for seq in rsa_sequences for value in seq]

        # 如果没有有效数值，返回原始序列
        if not all_values:
            print("没有有效的数值，无法归一化。")
            return rsa_sequences

        min_val, max_val = min(all_values), max(all_values)

        # 如果最大值等于最小值，则所有数值归一化为0.5
        if min_val == max_val:
            print(f"所有数值相同：{min_val}，归一化为0.5。")
            return [[0.5 for value in seq] for seq in rsa_sequences]

        # 进行归一化处理
        normalized_rsa = []
        for rsa_seq in rsa_sequences:
            normalized_seq = [(value - min_val) / (max_val - min_val) for value in rsa_seq]
            normalized_rsa.append(normalized_seq)

        return normalized_rsa

    # 填充或截断序列以适应 max_length
    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                padded_sequences.append(seq[:max_length])
            else:
                padded_sequences.append(seq + [0] * (max_length - len(seq)))
        return padded_sequences

    # 数据集长度
    def __len__(self):
        return len(self.aa_sequences)

    # 取出一个样本
    def __getitem__(self, idx):
        aa_seq = torch.tensor(self.aa_sequences[idx], dtype=torch.long)
        struct_seq = torch.tensor(self.structure_sequences[idx], dtype=torch.long)
        rsa_seq = torch.tensor(self.rsa_sequences[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (aa_seq, struct_seq,rsa_seq), label


class ARGDataset_multi_channel2(Dataset):
    def __init__(self, aa_sequences, structure_sequences, labels, max_length):
        """
        参数:
        aa_sequences: 氨基酸序列列表
        structure_sequences: 二级结构序列列表
        labels: 标签列表
        max_length: 序列的最大长度
        """
        self.aa_sequences = self.pad_sequences(self.tokenize_sequences(aa_sequences, 'aa'), max_length)
        self.structure_sequences = self.pad_sequences(self.tokenize_sequences(structure_sequences, 'structure'), max_length)
        self.labels = labels

    # 将氨基酸和结构序列转化为数字序列
    def tokenize_sequences(self, sequences, mode='aa'):
        """
        mode = 'aa': 处理氨基酸序列
        mode = 'structure': 处理二级结构序列
        """
        if mode == 'aa':
            amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'  # 假设定义了氨基酸字典
            tokenizer = {aa: idx+1 for idx, aa in enumerate(amino_acids)}  # 0保留给 padding
        elif mode == 'structure':
            structures = 'HGIEBTSC'  # H=α螺旋, G=310螺旋,I-π-螺旋,E-延伸的 β-链,B-孤立的 β-桥 ,T-转角,S-弯曲,C-无规卷曲
            tokenizer = {struct: idx+1 for idx, struct in enumerate(structures)}  # 0保留给 padding
        else:
            raise ValueError("Invalid mode, use 'aa' or 'structure'.")

        def tokenize_sequence(sequence):
            return [tokenizer[char] for char in sequence if char in tokenizer]

        return [tokenize_sequence(seq) for seq in sequences]


    # 填充或截断序列以适应 max_length
    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                padded_sequences.append(seq[:max_length])
            else:
                padded_sequences.append(seq + [0] * (max_length - len(seq)))
        return padded_sequences


    # 数据集长度
    def __len__(self):
        return len(self.aa_sequences)

    # 取出一个样本
    def __getitem__(self, idx):
        aa_seq = torch.tensor(self.aa_sequences[idx], dtype=torch.long)
        struct_seq = torch.tensor(self.structure_sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (aa_seq, struct_seq), label
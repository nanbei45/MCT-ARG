import csv
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
from Bio import SeqIO
from dataset import ARGDataset_multi_channel
from transformer import MultiChannelTransformerModel3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_prediction_data(aa_file, ss_file, rsa_file):
    aa_sequences = [str(record.seq) for record in SeqIO.parse(aa_file,"fasta")]
    ss_sequences = [str(record.seq) for record in SeqIO.parse(ss_file,"fasta")]
    # 解析RSA序列
    rsa_sequences = []
    with open(rsa_file) as f:
        lines = [l.strip() for l in f]
    rsa_seq = []
    for line in lines:
        if line.startswith(">"):
            if rsa_seq:
                rsa_sequences.append(rsa_seq)
                rsa_seq = []
        else:
            rsa_seq = list(map(int, line.split()))
    if rsa_seq:
        rsa_sequences.append(rsa_seq)

        # 保持原始断言检查
    assert len(aa_sequences) == len(ss_sequences) == len(rsa_sequences), "输入文件样本数不一致"

    return aa_sequences, ss_sequences, rsa_sequences


def load_model(model_path):
    """加载预训练模型"""
    model = MultiChannelTransformerModel3(
        aa_input_dim=22,
        struct_input_dim=9,
        rsa_input_dim=22,
        model_dim=64,
        num_heads=8,
        num_layers=2,
        ff_dim=128,
        num_classes=2,
        max_seq_len=512
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_dataset(aa_sequences, ss_sequences, rsa_sequences, max_length=512):
    """创建预测数据集"""
    # 生成伪标签
    labels = [3] * len(aa_sequences)
    return ARGDataset_multi_channel(
        aa_sequences=aa_sequences,
        structure_sequences=ss_sequences,
        rsa_sequences=rsa_sequences,
        labels=labels,
        max_length=max_length
    )


def get_sequence_headers(file_path):
    """从fasta文件中提取序列头信息"""
    headers = []
    with open(file_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].split("|")[0].strip()
                headers.append(header)
    return headers


def save_attention_embedding(model, dataset, output_dir, batch_size=32):
    """执行预测并保存结果"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []
    attention_data = []
    valid_lengths = dataset.aa_valid_lengths
    embedding_data = []

    with torch.no_grad():
        for batch_idx, ((aa, struct, rsa), _) in enumerate(dataloader):
            aa = aa.to(device)
            struct = struct.to(device)
            rsa = rsa.to(device)

            outputs, attn_dict = model(aa, struct, rsa)
            #outputs = model(aa, struct, rsa)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            batch_attn = {
                "aa": attn_dict["aa_attentions"][0].cpu().numpy(),
                "struct": attn_dict["struct_attentions"][0].cpu().numpy(),
                "rsa": attn_dict["rsa_attentions"][0].cpu().numpy()
            }


            batch_embed = {
                "aa": attn_dict["aa_embedding"].cpu().numpy(),  # shape: (B, L, D)
                "struct": attn_dict["struct_embedding"].cpu().numpy(),
                "rsa": attn_dict["rsa_embedding"].cpu().numpy()
            }

            for i in range(aa.size(0)):
                sample_idx = batch_idx * batch_size + i
                valid_len = valid_lengths[sample_idx]

                attn_record = {
                    "aa": batch_attn["aa"][i][:, :valid_len, :valid_len],
                    "struct": batch_attn["struct"][i][:, :valid_len, :valid_len],
                    "rsa": batch_attn["rsa"][i][:, :valid_len, :valid_len]
                }
                attention_data.append(attn_record)

                embedding_record = {
                    "aa": batch_embed["aa"][i][:valid_len],
                    "struct": batch_embed["struct"][i][:valid_len],
                    "rsa": batch_embed["rsa"][i][:valid_len]
                }
                embedding_data.append(embedding_record)

    os.makedirs(output_dir, exist_ok=True)

    attn_dir = os.path.join(output_dir, "attention_weights")
    os.makedirs(attn_dir, exist_ok=True)

    embedding_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    for idx, (attn,emb) in enumerate(zip(attention_data, embedding_data)):
        np.savez_compressed(
            os.path.join(attn_dir, f"OXA-48_{idx}.npz"),
            aa_attn=attn["aa"],
            struct_attn=attn["struct"],
            rsa_attn=attn["rsa"]
        )

        np.savez_compressed(
            os.path.join(embedding_dir, f"OXA-48_{idx}.npz"),
            aa_emb=emb["aa"],
            struct_emb=emb["struct"],
            rsa_emb=emb["rsa"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGs prediction script")
    parser.add_argument("-a", "--aa_file", default="test/test.fasta", help="Amino acid sequence file path")
    parser.add_argument("-s", "--ss_file", default="test/test.ss8", help="Secondary structure file path")
    parser.add_argument("-r", "--rsa_file", default="test/test.acc20", help="RSA file path")
    parser.add_argument("--model_path", default="model/BinaryClassificationModel/best_global_model.pth",
                        help="model_path")
    parser.add_argument("-o", "--output_dir", default="output", help="out_dir")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")

    args = parser.parse_args()

    aa_seqs, ss_seqs, rsa_seqs = parse_prediction_data(args.aa_file, args.ss_file, args.rsa_file)
    dataset = create_dataset(aa_seqs, ss_seqs, rsa_seqs)
    model = load_model(args.model_path)

    save_attention_embedding(model, dataset, args.output_dir, args.batch_size)


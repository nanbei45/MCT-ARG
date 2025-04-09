import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
from Bio import SeqIO

from dataset import ARGDataset_multi_channel
from transformer import MultiChannelTransformerModel3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_prediction_data1(aa_file, ss_file, rsa_file):
    aa_sequences, ss_sequences, rsa_sequences = [], [], []
    with open(aa_file) as f:
        lines = [l.strip() for l in f]
    aa_seq = []
    for line in lines:
        if line.startswith(">"):
            if aa_seq:
                aa_sequences.append("".join(aa_seq))
                aa_seq = []
        else:
            aa_seq.append(line)
    if aa_seq:
        aa_sequences.append("".join(aa_seq))
    with open(ss_file) as f:
        lines = [l.strip() for l in f]
    ss_seq = []
    for line in lines:
        if line.startswith(">"):
            if ss_seq:
                ss_sequences.append("".join(ss_seq))
                ss_seq = []
        else:
            ss_seq.append(line)
    if ss_seq:
        ss_sequences.append("".join(ss_seq))

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

    assert len(aa_sequences) == len(ss_sequences) == len(rsa_sequences), "The number of samples in the input file is inconsistent"
    return aa_sequences, ss_sequences, rsa_sequences

def parse_prediction_data(aa_file, ss_file, rsa_file):
    aa_sequences = [str(record.seq) for record in SeqIO.parse(aa_file,"fasta")]
    ss_sequences = [str(record.seq) for record in SeqIO.parse(ss_file,"fasta")]
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

    assert len(aa_sequences) == len(ss_sequences) == len(rsa_sequences), "The number of samples in the input file is inconsistent"

    return aa_sequences, ss_sequences, rsa_sequences


def load_model(model_path):
    model = MultiChannelTransformerModel3(
        aa_input_dim=22,
        struct_input_dim=22,
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
    labels = [0] * len(aa_sequences)
    return ARGDataset_multi_channel(
        aa_sequences=aa_sequences,
        structure_sequences=ss_sequences,
        rsa_sequences=rsa_sequences,
        labels=labels,
        max_length=max_length
    )


def get_sequence_headers(file_path):
    headers = []
    with open(file_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].split("|")[0].strip()
                headers.append(header)
    return headers


def predict_and_save(model, dataset, output_dir, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []

    with torch.no_grad():
        for batch_idx, ((aa, struct, rsa), _) in enumerate(dataloader):
            aa = aa.to(device)
            struct = struct.to(device)
            rsa = rsa.to(device)

            outputs, attn_dict = model(aa, struct, rsa)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())



    os.makedirs(output_dir, exist_ok=True)
    headers = get_sequence_headers(args.aa_file)

    results = pd.DataFrame({
        "ID": headers,
        "Prediction": all_preds,
        "Probability_0": [p[0] for p in all_probs],
        "Probability_1": [p[1] for p in all_probs]
    })

    results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGs预测脚本")
    parser.add_argument("-a","--aa_file",  default="test/test.fasta",help="氨基酸序列文件路径")
    parser.add_argument("-s","--ss_file", default="test/test.ss8", help="二级结构文件路径")
    parser.add_argument("-r","--rsa_file",  default="test/test.acc20",help="RSA文件路径")
    parser.add_argument("--model_path", default="model/BinaryClassificationModel/best_global_model.pth", help="model_path")
    parser.add_argument("-o","--output_dir", default="output", help="out_dir")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    args = parser.parse_args()

    aa_seqs, ss_seqs, rsa_seqs = parse_prediction_data(args.aa_file, args.ss_file, args.rsa_file)
    dataset = create_dataset(aa_seqs, ss_seqs, rsa_seqs)
    model = load_model(args.model_path)

    predict_and_save(model, dataset, args.output_dir, args.batch_size)
    print(f"Prediction completed! Results saved to {args.output_dir}")

import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
import argparse

from dataset import parse_data, ARGDataset, ARGDataset_multi_channel
from transformer import MultiChannelTransformerModel3, AECRLoss
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, roc_curve, \
    precision_recall_curve

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def parse_data_multi_channel(aa_file_path,struct_file_path,rsa_file_path):
    labels = []
    aa_sequences = []
    struct_sequences = []
    rsa_sequences = []
    with open(aa_file_path, 'r') as aa_file:
        for line in aa_file:
            if line.startswith('>'):
                parts = line[1:].strip().split('|')
                label = int(parts[-1])
                labels.append(label)
            else:
                aa_seq = line[:-1]
                aa_sequences.append(aa_seq)
    with open(struct_file_path, 'r') as struct_file:
        for line in struct_file:
            if line.startswith('>'):
                pass
            else:
                struct_seq = line[:-1]
                struct_sequences.append(struct_seq)
    with open(rsa_file_path, 'r') as rsa_file:
        for line in rsa_file:
            if line.startswith('>'):
                pass
            else:
                rsa_sequences.append([int(x) for x in line.strip().split()])
    return np.array(labels),np.array(aa_sequences),np.array(struct_sequences),np.array(rsa_sequences,dtype=object)

def load_fasta_files(aa_file_path,struct_file_path,rsa_file_path):
    aa_sequences = []
    struct_sequences = []
    rsa_sequences = []
    labels = []
    with open(aa_file_path, 'r') as aa_file:
        for line in aa_file:
            if line.startswith('>'):
                parts = line[1:].strip().split('|')
                label = int(parts[-1])
                labels.append(label)
            else:
                aa_seq = line[:-1]
                aa_sequences.append(aa_seq)
    for record in SeqIO.parse(struct_file_path, "fasta"):
        struct_sequences.append(str(record.seq))
    with open(rsa_file_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                rsa_sequences.append([int(x) for x in line.strip().split()])
    return (
        np.array(aa_sequences),
        np.array(struct_sequences),
        np.array(rsa_sequences, dtype=object),
        np.array(labels)
    )


def split_dataset_multi_channel(aa_sequences,struct_sequences, rsa_sequences,labels, train_size=0.8, val_size=0.2, test_size=0.1):
    train_aa_sequences, val_aa_sequences,train_struct_sequences, val_struct_sequences,train_rsa_sequences, val_rsa_sequences, train_labels,val_labels = train_test_split(
        aa_sequences,struct_sequences, rsa_sequences,labels, train_size=train_size, random_state=42)

    return train_aa_sequences,train_struct_sequences,train_rsa_sequences, train_labels, val_aa_sequences,val_struct_sequences,val_rsa_sequences, val_labels
def create_dataloader_multi_channel(aa_sequences,struct_sequences, rsa_sequences,labels, max_length=512, batch_size=64):

    dataset = ARGDataset_multi_channel(aa_sequences, struct_sequences,rsa_sequences,labels, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0
    aecr_loss = AECRLoss()
    for (aa_inputs, struct_inputs, rsa_inputs), labels in train_loader:
        aa_inputs, struct_inputs, rsa_inputs, labels = aa_inputs.to(device), struct_inputs.to(device), rsa_inputs.to(
            device), labels.to(device)
        outputs, attentions_dict = model(aa_inputs, struct_inputs, rsa_inputs)

        optimizer.zero_grad()

        task_loss = criterion(outputs, labels)
        reg_loss = aecr_loss(attentions_dict)
        loss = task_loss + reg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    return total_loss / len(train_loader), precision, recall

def test_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for (aa_inputs, struct_inputs, rsa_inputs), labels in test_loader:
            aa_inputs, struct_inputs, rsa_inputs, labels = (
                aa_inputs.to(device),
                struct_inputs.to(device),
                rsa_inputs.to(device),
                labels.to(device),
            )
            outputs, _ = model(aa_inputs, struct_inputs, rsa_inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    precision = precision_score(all_labels, all_preds,average='binary')
    recall = recall_score(all_labels, all_preds,average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels,all_preds)
    return precision, recall, auc, correct/total,mcc,f1


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--is_new_data', type=bool, default=False, help='default=False')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers, you had better put it '
                             '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training, default=64')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for, default=10')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Select the learning rate, default=1e-3')
    parser.add_argument('--input_dim', type=int, default=22, help='Types of Amino Acids, default=22')
    parser.add_argument('--model_dim', type=int, default=64, help='Hidden layer dimensions of Transformer, default=64')
    parser.add_argument('--num_layers', type=int, default=2, help=', default=2')
    parser.add_argument('--num_heads', type=int, default=8, help='The number of heads in the multi-head attention mechanism, default=8')
    parser.add_argument('--ff_dim', type=int, default=128,help='Dimensions of a feedforward network, default=128')
    parser.add_argument("--num_classes", type=int, default=2, help='Number of categories, default=2')
    parser.add_argument("--max_seq_len", type=int, default=512, help='Maximum length of amino acid sequence, default=500')
    parser.add_argument("--lambda_", type=int, default=10, help='Parameters of ewc,default=10')
    parser.add_argument("--expansion_threshold", type=float, default=0.01, help='Dynamic expansion threshold, default=0.01')
    parser.add_argument("--max_layers",type=int,default=12,help='Dynamically increasing number of layers,default=12')
    parser.add_argument("-a",'--aa_train_data_path', type=str, default='classified_sequences/all_sequences2.fasta', help='Path to the train dataset file')
    parser.add_argument("-s",'--struct_train_data_path', type=str, default='classified_sequences/all_sequences2.out.ss8',
                        help='Path to the train dataset file')
    parser.add_argument("-r",'--rsa_train_data_path', type=str, default='classified_sequences/all_sequences2.out.acc20',
                        help='Path to the train dataset file')
    parser.add_argument('--test_data_path', type=str, default='Is_ARGs/all_test.fasta', help='Path to the test dataset file')
    parser.add_argument('--new_data_path', type=str, default='Is_ARGs/all_new.fasta', help='Path to the new dataset file')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data used for training')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data used for validation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data used for testing')
    parser.add_argument('--output_dir', type=str, default='model', help='Directory to save trained model')
    return parser.parse_args()

best_model_path = ''
def main():
    # 获取参数
    args = parse_args()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    results = []
    early_stopping_patience = 20
    best_global_auc = 0.0  #
    best_global_model_path = 'model/BinaryClassificationModel/best_global_model.pth'

    labels, aa_sequences, struct_sequences,rsa_sequences = parse_data_multi_channel(args.aa_train_data_path,
                                                                      args.struct_train_data_path,args.rsa_train_data_path)
    train_aa_sequences, train_struct_sequences, train_rsa_sequences, train_labels, val_aa_sequences, val_struct_sequences, val_rsa_sequences, val_labels = split_dataset_multi_channel(
        aa_sequences, struct_sequences, rsa_sequences, labels, args.train_size, args.val_size, args.test_size)

    fold_metrics = {
        'Precision': [],
        'Recall': [],
        'AUC': [],
        'Accuracy': [],
        'MCC': [],
        'F1': []
    }

    for train_index, val_index in kf.split(train_aa_sequences):
        fold += 1
        best_epoch = 0
        print(f'Fold{fold}')
        fold_best_model_path = f'best_fold{fold}_model.pth'
        no_improve_epochs = 0
        best_fold_precision, best_fold_recall, best_fold_auc, best_fold_correct, best_fold_mcc, best_fold_f1 = 0.0,0.0,0.0,0.0,0.0,0.0
        cv_train_aa_sequences = [train_aa_sequences[i] for i in train_index]
        cv_train_struct_sequences = [train_struct_sequences[i] for i in train_index]
        cv_train_rsa_sequences = [train_rsa_sequences[i] for i in train_index]
        cv_train_labels = [train_labels[i] for i in train_index]
        cv_val_aa_sequences = [train_aa_sequences[i] for i in val_index]
        cv_val_struct_sequences = [train_struct_sequences[i] for i in val_index]
        cv_val_rsa_sequences = [train_rsa_sequences[i] for i in val_index]
        cv_val_labels = [train_labels[i] for i in val_index]

        train_loader = create_dataloader_multi_channel(cv_train_aa_sequences, cv_train_struct_sequences,
                                                       cv_train_rsa_sequences, cv_train_labels, args.max_seq_len,
                                                       args.batch_size)
        val_loader = create_dataloader_multi_channel(cv_val_aa_sequences, cv_val_struct_sequences, cv_val_rsa_sequences,
                                                     cv_val_labels,
                                                     args.max_seq_len, args.batch_size)
        model = MultiChannelTransformerModel3(args.input_dim,args.input_dim, args.input_dim,args.model_dim, args.num_heads, args.num_layers, args.ff_dim, args.num_classes, args.max_seq_len).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(args.num_epochs):
            train_loss, train_precision, train_recall = train_model(train_loader, model, criterion, optimizer,
                                                                args.num_epochs)
            val_precision, val_recall,val_auc, val_correct,val_mcc,val_f1 = test_model(val_loader, model)
            print(f'Epoch {epoch + 1}/{args.num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}')
            print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f},Val AUC: {val_auc:.4f}, Val Accuracy: {val_correct:.4f}, Val mcc: {val_mcc:.4f}, Val F1: {val_f1:.4f}')

            if val_auc > best_fold_auc:
                best_fold_precision, best_fold_recall, best_fold_auc, best_fold_correct, best_fold_mcc, best_fold_f1 = val_precision, val_recall,val_auc, val_correct,val_mcc,val_f1
                torch.save(model.state_dict(), fold_best_model_path)
                best_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'Fold {fold}: Best model found at epoch {best_epoch} with acc: {best_fold_auc:.4f}')
                print(
                    f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f},Val AUC: {val_auc:.4f}, Val Accuracy: {val_correct:.4f}, Val mcc: {val_mcc:.4f}, Val F1: {val_f1:.4f}')

                break

        if best_fold_auc > best_global_auc:
            best_global_auc = best_fold_auc
            torch.save(model.state_dict(), best_global_model_path)

        results.append([fold, best_fold_precision, best_fold_recall, best_fold_auc, best_fold_correct, best_fold_mcc, best_fold_f1])

        fold_metrics['Precision'].append(best_fold_precision)
        fold_metrics['Recall'].append(best_fold_recall)
        fold_metrics['AUC'].append(best_fold_auc)
        fold_metrics['Accuracy'].append(best_fold_correct)
        fold_metrics['MCC'].append(best_fold_mcc)
        fold_metrics['F1'].append(best_fold_f1)
        print(f'Fold {fold}: Best model found at epoch {best_epoch} with acc: {best_fold_auc:.4f}')
    save_results_to_csv(results, 'result/IsARGs/result.csv')

def save_results_to_csv(results, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Precision', 'Recall','AUC', 'Accuracy','MCC',"F1"])
        for result in results:
            writer.writerow(result)
    print(f'Results saved to {file_name}')

if __name__ == '__main__':
    main()
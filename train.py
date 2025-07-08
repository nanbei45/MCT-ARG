import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from sklearn.metrics import precision_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score
from torch.utils.data import DataLoader
import argparse

from dataset import ARGDataset_multi_channel
from transformer import MultiChannelTransformerModel3, AECRLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

def load_dataset_from_dir(data_dir,type):
    aa_sequences = []
    struct_sequences = []
    rsa_sequences = []
    labels = []

    aa = 'protein'
    ss = 'ss'
    rsa = 'acc20'

    categories = sorted([d for d in os.listdir(os.path.join(data_dir,aa,type))])
    print(categories)
    category_to_index = {category: idx for idx, category in enumerate(categories)}
    print(category_to_index)

    aa_dir = os.path.join(data_dir,aa,type)
    aa_files = [f for f in os.listdir(aa_dir) if f.endswith('.fasta')]

    for aa_file in aa_files:
        aa_path = os.path.join(aa_dir, aa_file)
        for record in SeqIO.parse(aa_path, 'fasta'):
            aa_sequences.append(str(record.seq))
            labels.append(category_to_index[aa_file])


    ss_dir = os.path.join(data_dir,ss,type)
    ss_files = [f for f in os.listdir(ss_dir) if f.endswith('.fasta')]

    for ss_file in ss_files:
        ss_path = os.path.join(ss_dir, ss_file)
        for record in SeqIO.parse(ss_path, 'fasta'):
            struct_sequences.append(str(record.seq))

    acc20_dir = os.path.join(data_dir,rsa,type)
    acc20_files = [f for f in os.listdir(acc20_dir) if f.endswith('.fasta')]

    for acc20_file in acc20_files:
        rsa_path = os.path.join(acc20_dir, acc20_file)
        with open(rsa_path, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    rsa_sequences.append([int(x) for x in line.strip().split()])

    assert len(aa_sequences) == len(struct_sequences) == len(rsa_sequences) == len(labels), \


    return (
        np.array(aa_sequences),
        np.array(struct_sequences),
        np.array(rsa_sequences, dtype=object),
        np.array(labels),
        categories
    )



def create_dataloader(aa_sequences, struct_sequences, rsa_sequences, labels, max_length=512, batch_size=64):
    dataset = ARGDataset_multi_channel(aa_sequences, struct_sequences, rsa_sequences, labels, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    aecr_loss = AECRLoss()
    for (aa_inputs, struct_inputs, rsa_inputs), labels in train_loader:
        aa_inputs = aa_inputs.to(device)
        struct_inputs = struct_inputs.to(device)
        rsa_inputs = rsa_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs,attentions_dict = model(aa_inputs, struct_inputs, rsa_inputs)
        task_loss = criterion(outputs, labels)
        reg_loss = aecr_loss(attentions_dict)  # 传入注意力字典
        loss = task_loss + reg_loss
        #loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    return total_loss / len(train_loader), precision, recall



def evaluate_model(loader, model, num_classes, save_predictions_path=None):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for (aa_inputs, struct_inputs, rsa_inputs), labels in loader:
            aa_inputs = aa_inputs.to(device)
            struct_inputs = struct_inputs.to(device)
            rsa_inputs = rsa_inputs.to(device)
            labels = labels.to(device)

            outputs,_ = model(aa_inputs, struct_inputs, rsa_inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    all_labels_one_hot = np.eye(num_classes)[all_labels]

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = correct / total

    try:
        auc = roc_auc_score(all_labels_one_hot, np.array(all_probs), average='weighted', multi_class='ovr')
    except ValueError:
        auc = np.nan

    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    if save_predictions_path:
        with open(save_predictions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['True_Label', 'Pred_Label'] + [f'Prob_Class_{i}' for i in range(num_classes)]
            writer.writerow(header)
            for label, pred, prob in zip(all_labels, all_preds, all_probs):
                writer.writerow([label, pred] + list(prob))

        print(f"Saved detailed test predictions to: {save_predictions_path}")

    return precision, recall, auc, accuracy, mcc, f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--input_dim', type=int, default=22, help='amino acid types')
    parser.add_argument('--model_dim', type=int, default=64, help='transformer hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=128, help='feedforward dimension')
    parser.add_argument('--max_seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--data_dir', type=str, default='E:\Code\ARG_redo\data_and_process\split_data', help='base data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory for results and models')
    return parser.parse_args()


def save_results_to_csv(results, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Precision', 'Train_Recall',
                         'Val_Precision', 'Val_Recall', 'Val_AUC', 'Val_Accuracy', 'Val_MCC', 'Val_F1'])
        for result in results:
            writer.writerow(result)
    print(f'Results saved to {file_name}')


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading training data...")
    train_aa, train_struct, train_rsa, train_labels, categories = load_dataset_from_dir(args.data_dir, 'train')

    print("Loading validation data...")
    val_aa, val_struct, val_rsa, val_labels, _ = load_dataset_from_dir(args.data_dir, 'val')

    print("Loading test data...")
    test_aa, test_struct, test_rsa, test_labels, _ = load_dataset_from_dir(args.data_dir, 'test')

    num_classes = len(categories)
    print(f"Loaded datasets with {num_classes} classes")
    print(f"Training samples: {len(train_aa)}")
    print(f"Validation samples: {len(val_aa)}")
    print(f"Test samples: {len(test_aa)}")


    train_loader = create_dataloader(
        train_aa, train_struct, train_rsa, train_labels,
        args.max_seq_len, args.batch_size
    )

    val_loader = create_dataloader(
        val_aa, val_struct, val_rsa, val_labels,
        args.max_seq_len, args.batch_size
    )

    test_loader = create_dataloader(
        test_aa, test_struct, test_rsa, test_labels,
        args.max_seq_len, args.batch_size
    )

    model = MultiChannelTransformerModel3(
        args.input_dim, 9,args.input_dim,
        args.model_dim, args.num_heads, args.num_layers,
        args.ff_dim, num_classes, args.max_seq_len
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = 20
    results = []


    for epoch in range(args.num_epochs):

        train_loss, train_precision, train_recall = train_model(train_loader, model, criterion, optimizer)


        val_precision, val_recall, val_auc, val_acc, val_mcc, val_f1 = evaluate_model(
            val_loader, model, num_classes
        )


        results.append([
            epoch + 1, train_loss, train_precision, train_recall,
            val_precision, val_recall, val_auc, val_acc, val_mcc, val_f1
        ])

        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}')
        print(
            f'Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f}')


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stopping_counter = 0
            model_path = os.path.join(args.output_dir, 'new_regularization_best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model at epoch {best_epoch} with val acc: {best_val_acc:.4f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


    print("\nTesting best model...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'new_regularization_best_model.pth')))

    test_pred_path = os.path.join(args.output_dir, 'new_regularization_test_predictions.csv')
    test_precision, test_recall, test_auc, test_acc, test_mcc, test_f1 = evaluate_model(
        test_loader, model, num_classes,test_pred_path
    )

    print("\nFinal Test Results:")
    print(f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f}')
    print(f'AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f}')
    print(f'MCC: {test_mcc:.4f} | F1: {test_f1:.4f}')

    results.append([
        'Test', '', '', '',
        test_precision, test_recall, test_auc, test_acc, test_mcc, test_f1
    ])

    save_results_to_csv(results, os.path.join(args.output_dir, 'new_regularization_training_results.csv'))

    test_results_path = os.path.join(args.output_dir, 'new_regularization_test_results.csv')
    with open(test_results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', test_precision])
        writer.writerow(['Recall', test_recall])
        writer.writerow(['AUC', test_auc])
        writer.writerow(['Accuracy', test_acc])
        writer.writerow(['MCC', test_mcc])
        writer.writerow(['F1', test_f1])

    print(f"Test results saved to {test_results_path}")


if __name__ == '__main__':
    main()

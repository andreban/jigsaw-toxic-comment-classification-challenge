import json
import torch

def load_dataset(filename):
    full_dataset = []
    with open(filename, mode='r') as dataset_file:
        for sample in dataset_file:
            sample = json.loads(sample)
            input = torch.tensor(sample['embeddings'])
            label = torch.tensor([
                float(sample['toxic']),
                float(sample['severe_toxic']),
                float(sample['obscene']),
                float(sample['threat']),
                float(sample['insult']),
                float(sample['identity_hate'])
            ])
            full_dataset.append((input, label))
    return full_dataset

def split_dataset(full_dataset, text_split_pct = 0.2):
    num_samples = len(full_dataset)
    num_validation = int(text_split_pct * num_samples)
    shuffled_indices = torch.randperm(num_samples)
    train_indices = shuffled_indices[:-num_validation]
    validation_indices = shuffled_indices[-num_validation:]

    training_dataset = [full_dataset[i] for i in train_indices]
    validation_dataset = [full_dataset[i] for i in validation_indices]

    return (training_dataset, validation_dataset)
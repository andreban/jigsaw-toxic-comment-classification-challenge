import torch
import torch.nn as nn
import torch.optim as optim

from text_embedding_004.toxicity_model import ToxicityModel
from toxicity_training import train_model

torch.manual_seed(1)

model = ToxicityModel()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = nn.BCEWithLogitsLoss()

train_model(
    input_filename='YT-comments-text-embedding-004-768-10k.jsonl',
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_size=512,
    epochs=100
)

torch.save(model.state_dict(), 'YT-comments-text-embedding-004-768-10k.safetensors')
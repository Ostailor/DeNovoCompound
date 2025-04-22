import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from smiles_generator.model import SmilesVAE, vae_loss
from smiles_generator.utils import load_smiles, Tokenizer
from smiles_generator.config import *


def train_vae_cli(data_path, epochs):
    smiles    = load_smiles(data_path)
    tokenizer = Tokenizer.build(smiles)
    inputs    = torch.stack([tokenizer.encode(s) for s in smiles])
    dataset   = TensorDataset(inputs, inputs)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SmilesVAE(len(tokenizer.itos), EMB_DIM, HID_DIM, Z_DIM, tokenizer.pad_idx).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR_VAE)

    for ep in range(1, epochs+1):
        total_loss = 0
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, mu, logvar = model(x, tokenizer.sos_idx, teacher_forcing=y)
            loss, rec, kl     = vae_loss(logits, y, mu, logvar, tokenizer.pad_idx)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep} | Loss: {total_loss/len(loader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'vae.pt'))
    with open(os.path.join(MODEL_DIR, 'vocab.json'), 'w') as f:
        import json; json.dump(tokenizer.itos, f)
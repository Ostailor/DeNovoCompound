import os
import torch
import torch.nn.functional as F
from smiles_generator.model import SmilesVAE
from smiles_generator.utils import Tokenizer, qed_reward, logp_reward, sa_reward
from smiles_generator.config import *


def finetune_cli(reward_list, num_steps):
    # Load vocab & model
    import json
    with open(os.path.join(MODEL_DIR,'vocab.json')) as f:
        itos = json.load(f)
    stoi     = {ch:i for i,ch in enumerate(itos)}
    tokenizer= Tokenizer(stoi, itos, max_len=len(itos))
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = SmilesVAE(len(itos), EMB_DIM, HID_DIM, Z_DIM, tokenizer.pad_idx).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,'vae.pt')))
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=LR_RL)

    # Mapping from name to function
    reward_fns = {'qed': qed_reward, 'logp': logp_reward, 'sa': sa_reward}

    for step in range(1, num_steps+1):
        # Sample latent
        z      = torch.randn(BATCH_SIZE, Z_DIM, device=device)
        logits = model.decode(z, tokenizer.max_len, tokenizer.sos_idx)
        logp   = F.log_softmax(logits, dim=-1)
        ids    = logp.argmax(dim=-1)
        logp_seq = logp.gather(2, ids.unsqueeze(-1)).sum((1,2))

        smiles   = tokenizer.decode_batch(ids)
        # Compute each reward
        rs = []
        for name in reward_list:
            fn = reward_fns[name]
            rs.append(fn(smiles).to(device))
        # Weighted composite reward
        total_r = sum(REWARD_WEIGHTS[name] * r for name, r in zip(reward_list, rs))

        loss = -(total_r * logp_seq).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 0:
            avg_r = total_r.mean().item()
            print(f"Step {step}/{num_steps} | Avg reward: {avg_r:.3f}")

    # Save fine-tuned
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'vae_finetuned.pt'))
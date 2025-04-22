import torch
from rdkit import Chem
from collections import Counter
from rdkit.Chem.QED import qed
from rdkit.Chem import Crippen, rdMolDescriptors

class Tokenizer:
    """
    Char‑level SMILES tokenizer with special tokens.
    """
    def __init__(self, stoi, itos, max_len):
        self.stoi    = stoi
        self.itos    = itos
        self.pad_idx = stoi['<pad>']
        self.sos_idx = stoi['<bos>']
        self.eos_idx = stoi['<eos>']
        self.max_len = max_len

    @classmethod
    def build(cls, smiles_list):
        tokens = Counter()
        for smi in smiles_list:
            tokens.update(list(smi))
        itos   = ['<pad>','<bos>','<eos>'] + sorted(tokens)
        stoi   = {ch:i for i,ch in enumerate(itos)}
        max_len= max(len(s)+2 for s in smiles_list)
        return cls(stoi, itos, max_len)

    def encode(self, smiles):
        seq = [self.sos_idx] + [self.stoi[ch] for ch in smiles] + [self.eos_idx]
        seq += [self.pad_idx] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def decode(self, ids):
        chars = [self.itos[i] for i in ids
                 if i not in {self.sos_idx, self.eos_idx, self.pad_idx}]
        return ''.join(chars)

    def decode_batch(self, batch_ids):
        return [self.decode(ids.tolist()) for ids in batch_ids]


def load_smiles(path):
    smiles = []
    for line in open(path):
        smi = line.strip()
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue
        if Chem.Descriptors.MolWt(mol) > 500: continue
        smiles.append(Chem.MolToSmiles(mol, True))
    return smiles

# Reward functions

def qed_reward(smiles_list):
    return torch.tensor([qed(Chem.MolFromSmiles(s)) for s in smiles_list])


def logp_reward(smiles_list):
    return torch.tensor([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles_list])


def sa_reward(smiles_list):
    """
    Synthetic Accessibility (SA) score: lower= easier to synthesize.
    We invert (10 - SA) so higher reward = easier synthesis.
    """
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        scores.append(rdMolDescriptors.CalcSyntheticAccessibilityScore(mol))
    sa = torch.tensor(scores)
    return 10.0 - sa
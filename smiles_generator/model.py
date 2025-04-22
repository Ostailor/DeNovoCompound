import torch
import torch.nn as nn
import torch.nn.functional as F

class SmilesVAE(nn.Module):
    """
    Variational Autoencoder for SMILES:
    Encodes token sequences to latent z and decodes z to SMILES logits.
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, z_dim, pad_idx):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder_rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.to_mu       = nn.Linear(hid_dim, z_dim)
        self.to_logvar   = nn.Linear(hid_dim, z_dim)
        self.decoder_rnn = nn.GRU(emb_dim + z_dim, hid_dim, batch_first=True)
        self.output_fc   = nn.Linear(hid_dim, vocab_size)

    def encode(self, x):
        emb   = self.embedding(x)
        _, h = self.encoder_rnn(emb)
        h     = h.squeeze(0)
        mu    = self.to_mu(h)
        logvar= self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len, sos_idx, teacher_forcing=None):
        B          = z.size(0)
        inp        = torch.full((B,1), sos_idx, dtype=torch.long, device=z.device)
        hidden     = None
        outputs    = []
        z_expanded = z.unsqueeze(1)
        for t in range(seq_len):
            emb     = self.embedding(inp)
            rnn_in  = torch.cat([emb, z_expanded], dim=2)
            out, h  = self.decoder_rnn(rnn_in, hidden)
            logits  = self.output_fc(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))
            if teacher_forcing is not None:
                inp = teacher_forcing[:, t].unsqueeze(1)
            else:
                inp = logits.argmax(dim=1).unsqueeze(1)
            hidden = h
        return torch.cat(outputs, dim=1)

    def forward(self, x, sos_idx, teacher_forcing=None):
        mu, logvar    = self.encode(x)
        z             = self.reparameterize(mu, logvar)
        recon_logits  = self.decode(z, x.size(1), sos_idx, teacher_forcing)
        return recon_logits, mu, logvar


def vae_loss(recon_logits, target, mu, logvar, pad_idx):
    rec_loss = F.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        target.view(-1),
        ignore_index=pad_idx
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / target.size(0)
    return rec_loss + kl, rec_loss, kl
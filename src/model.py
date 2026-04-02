import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size = x.size(0)
        _, (h_n, _) = self.encoder(x)
        decoder_input = torch.zeros((batch_size, self.seq_len, 1), device=x.device)
        decoder_out, _ = self.decoder(decoder_input, (h_n, torch.zeros_like(h_n)))
        out = self.output_layer(decoder_out)
        return out


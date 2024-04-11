import torch
import torch.nn as nn

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update Gate
        self.Wz = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))
        
        # Reset Gate
        self.Wr = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        
        # Candidate Layer
        self.Wh = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, hidden_state):
        h_prev = hidden_state
        
        combined = torch.cat((x, h_prev), dim=1)
        z = torch.sigmoid(torch.matmul(combined, self.Wz) + self.bz)
        r = torch.sigmoid(torch.matmul(combined, self.Wr) + self.br)
        combined_reset = torch.cat((x, r * h_prev), dim=1)
        h_tilde = torch.tanh(torch.matmul(combined_reset, self.Wh) + self.bh)
        h = (1 - z) * h_prev + z * h_tilde
        
        return h

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        self.config_model = config['model']
        self.input_size = self.config_model['input_size']
        self.hidden_size = self.config_model['hidden_size']
        self.output_size = self.config_model['output_size']

        self.gru_cell = CustomGRUCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config_model['dropout_rate'])  # Dropout rate

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            h = self.gru_cell(x[:, t, :], h)

        h = self.dropout(h)
        out = self.fc(h)
        out = self.act(out)

        return out

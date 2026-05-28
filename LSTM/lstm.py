# lstm.py
import mindspore.nn as nn

class LSTMModel(nn.Cell):
    def __init__(self, input_size, hidden_size, dense_units, output_size, keep_prob=0.9):
        super().__init__()

        # --- HIDDEN LAYER LSTM ---
        # Layer LSTM 1: Menerima input awal. 
        # num_layers=2 akan menumpuk 2 layer LSTM secara vertikal
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(keep_prob=keep_prob)
        
        # --- HIDDEN LAYER DENSE ---
        # Layer Dense 1 (Hidden)
        self.fc1 = nn.Dense(hidden_size, dense_units)
        self.relu1 = nn.ReLU()
        
        # Layer Dense 2 (Hidden Tambahan) -> Mengurangi dimensi perlahan sebelum ke output
        self.fc2 = nn.Dense(dense_units, dense_units // 2)
        self.relu2 = nn.ReLU()
        
        # --- OUTPUT LAYER ---
        # Layer Output sekarang menerima input dari fc2 (dense_units // 2)
        self.fc3 = nn.Dense(dense_units // 2, output_size)

    def construct(self, x):
        # x, _ = self.lstm(x) akan otomatis memproses lewat 2 layer LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Ambil timestep terakhir

        # Poses lewat Hidden Layer Dense 1
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        
        # Proses lewat Hidden Layer Dense 2 (Tambahan)
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Proses ke Output Layer
        x = self.fc3(x)

        return x
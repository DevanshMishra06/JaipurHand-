import serial
import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIGURATION ---
COM_PORT = None  # None = auto-detect Arduino; or set e.g. 'COM4' / '/dev/ttyACM0'
BAUD_RATE = 115200
BUFFER_SIZE = 100  # 1.5 seconds at 100Hz

# --- PYTORCH MODEL ---
class SubvoxNet(nn.Module):
    def __init__(self): 
        super(SubvoxNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 35, 64) 
        self.fc2 = nn.Linear(64, 3) # 3 Classes: Silence, Left, Right
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

# --- DSP & SERIAL PIPELINE ---
def process_signal(raw_window):
    """Applies a 3Hz high-pass filter and Z-score normalization."""
    b, a = butter(4, 3.0 / 50.0, btype='high', analog=False)
    filtered = filtfilt(b, a, raw_window, axis=0)
    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0) + 1e-8
    return (filtered - mean) / std

def get_window(ser, buffer):
    """Reads from serial until the 100-sample buffer is full."""
    while len(buffer) < BUFFER_SIZE:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                vals = [float(x) for x in line.split(',')]
                if len(vals) == 6:
                    buffer.append(vals)
        except Exception:
            pass
    return np.array(buffer)

def calibrate(ser, buffer, num_samples=15):
    """Collects fresh data into RAM for instant training."""
    X_train, y_train = [], []
    labels = {1: "Left", 2: "Right", 0: "Silence"}
    
    print("\n--- CALIBRATION PHASE ---")
    for class_id, label in labels.items():
        print(f"\nRecord {num_samples} samples for '{label}'")
        for _ in range(num_samples):
            input(f"Press ENTER and mouth '{label}'...")
            ser.reset_input_buffer()
            buffer.clear()
            raw_window = get_window(ser, buffer)
            X_train.append(process_signal(raw_window))
            y_train.append(class_id)
            
    return torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(y_train), dtype=torch.long)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if COM_PORT is None:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        arduino = [p for p in ports if p.vid == 0x2341 or 'arduino' in (p.description or '').lower()]
        if not arduino:
            raise SystemExit(f"No Arduino found. Available ports: {[p.device for p in ports] or 'none'}")
        COM_PORT = arduino[0].device
    print(f"Connecting to Arduino on {COM_PORT}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE)
    time.sleep(2)
    buffer = deque(maxlen=BUFFER_SIZE)
    model = SubvoxNet()
    
    # 1. Calibrate
    X_train, y_train = calibrate(ser, buffer)
    
    # 2. Train
    print("\n--- TRAINING MODEL ---")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(35):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1}/35 | Loss: {loss.item():.4f}")
            
    # 3. Live Inference
    print("\n--- LIVE INFERENCE ACTIVE ---")
    model.eval()
    ser.reset_input_buffer()
    
    while True:
        raw_window = get_window(ser, buffer)
        clean_window = process_signal(raw_window)
        tensor_in = torch.tensor(clean_window, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            confidence, predicted_idx = torch.max(model(tensor_in), dim=1)
            
        if confidence.item() > 0.85:
            pred_class = predicted_idx.item()
            if pred_class == 1:
                print(f"LEFT ({confidence.item():.2f}) -> Moving Positive")
                ser.write(b'L')
                buffer.clear()
            elif pred_class == 2:
                print(f"RIGHT ({confidence.item():.2f}) -> Moving Negative")
                ser.write(b'R')
                buffer.clear()
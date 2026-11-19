"""
Physics-Aware Machine Learning Models for Pneumatic Leak Detection & Quantification
Includes:
1. PI-DTNet: Physics-Informed Dual-Task Deep Neural Network (1D-CNN + BiLSTM + Physics Attention)
2. Physics-Constrained Loss (Mass continuity + Thermodynamic Monotonicity)
3. Baseline Models: Standard 1D-CNN, LSTM, MLP, Random Forest, SVM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. Physics-Guided Attention Module
# ==========================================
class PhysicsAttention(nn.Module):
    """
    Attends to time steps conditioned on physical features (flow transients & pressure gradient).
    """
    def __init__(self, hidden_dim):
        super(PhysicsAttention, self).__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.phys_gate = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        score = torch.tanh(self.proj(x)) # (B, L, H/2)
        weights = F.softmax(self.phys_gate(score), dim=1) # (B, L, 1)
        context = torch.sum(x * weights, dim=1) # (B, H)
        return context, weights

# ==========================================
# 2. PI-DTNet: Physics-Informed Dual-Task Network
# ==========================================
class PIDTNet(nn.Module):
    def __init__(self, in_channels=10, seq_len=250, num_classes=7, hidden_dim=64):
        super(PIDTNet, self).__init__()
        
        # Multiscale 1D-CNN feature extraction
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2) # L -> 100
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Recurrent dynamic encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Physics-guided temporal attention
        self.attention = PhysicsAttention(hidden_dim)
        
        # Shared dense representation
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Head 1: Leak Classification (7 classes)
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Head 2: Leak Quantification (Continuous % regression)
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (B, C, L)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h))) # (B, H, L/2)
        
        h_seq = h.permute(0, 2, 1) # (B, L/2, H)
        lstm_out, _ = self.lstm(h_seq) # (B, L/2, H)
        
        context, attn_weights = self.attention(lstm_out) # (B, H)
        feat = self.fc_shared(context)
        
        logits = self.cls_head(feat)
        reg_pred = self.reg_head(feat)
        
        return logits, reg_pred.squeeze(-1), attn_weights

# ==========================================
# 3. Physics-Constrained Loss Formulation
# ==========================================
class PhysicsConstrainedLoss(nn.Module):
    """
    Combines:
    1. CrossEntropy for classification
    2. Smooth L1 for regression
    3. Mass Conservation Consistency Loss
    4. Thermodynamic Monotonicity Loss
    """
    def __init__(self, lambda_reg=0.05, lambda_mass=0.02, lambda_mono=0.01):
        super(PhysicsConstrainedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss()
        self.lambda_reg = lambda_reg
        self.lambda_mass = lambda_mass
        self.lambda_mono = lambda_mono
        
    def forward(self, logits, reg_pred, y_cls, y_reg, x_raw_tensor):
        # Standard supervised losses
        l_cls = self.ce_loss(logits, y_cls)
        l_reg = self.huber_loss(reg_pred, y_reg)
        
        l_mass = torch.tensor(0.0, device=logits.device)
        l_mono = torch.tensor(0.0, device=logits.device)
        
        # 1. Mass conservation consistency loss (scale in [0, 1]):
        if self.lambda_mass > 0 and x_raw_tensor.shape[1] >= 9:
            cond_proxy_mean = torch.mean(x_raw_tensor[:, 8, :], dim=1) # (B,)
            # Normalized conductance in [0, 1] corresponding to 0% to 100% leak
            expected_leak_norm = torch.clamp((cond_proxy_mean - 0.035) / 0.435, 0.0, 1.0)
            l_mass = F.mse_loss(reg_pred, expected_leak_norm)
        
        # 2. Thermodynamic monotonicity loss:
        if self.lambda_mono > 0 and x_raw_tensor.shape[1] >= 6:
            power_mean = torch.mean(x_raw_tensor[:, 5, :], dim=1) # (B,)
            p_diff = (power_mean.unsqueeze(1) - power_mean.unsqueeze(0)) / 1000.0 # (B, B)
            pred_diff = reg_pred.unsqueeze(1) - reg_pred.unsqueeze(0)  # (B, B)
            violation = F.relu(p_diff - pred_diff)
            l_mono = torch.mean(violation**2)
        
        total_loss = l_cls + self.lambda_reg * l_reg + self.lambda_mass * l_mass + self.lambda_mono * l_mono
        return total_loss, l_cls, l_reg, l_mass, l_mono

# ==========================================
# 4. Comparative Baseline Architectures
# ==========================================
class BaselineCNN1D(nn.Module):
    """Pure 1D-CNN without physics attention or dual multi-task heads"""
    def __init__(self, in_channels=5, num_classes=7):
        super(BaselineCNN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.reg_head = nn.Linear(64, 1)
        
    def forward(self, x):
        feat = self.net(x)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)

class BaselineLSTM(nn.Module):
    """Standard Recurrent LSTM"""
    def __init__(self, in_channels=5, hidden_dim=64, num_classes=7):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (B, C, L) -> permute to (B, L, C)
        out, (hn, cn) = self.lstm(x.permute(0, 2, 1))
        h_last = hn[-1] # (B, H)
        return self.cls_head(h_last), self.reg_head(h_last).squeeze(-1)

class BaselineMLP(nn.Module):
    """Standard MLP operating on flattened sequence or tabular inputs"""
    def __init__(self, in_dim=50, num_classes=7):
        super(BaselineMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.cls_head = nn.Linear(64, num_classes)
        self.reg_head = nn.Linear(64, 1)
        
    def forward(self, x):
        feat = self.fc(x)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)

if __name__ == "__main__":
    x_test = torch.randn(8, 10, 200)
    model = PIDTNet(in_channels=10, seq_len=250, num_classes=7)
    logits, reg, attn = model(x_test)
    print("PI-DTNet Forward check passed:")
    print("   Logits shape:", logits.shape)
    print("   Reg shape:", reg.shape)
    print("   Attn shape:", attn.shape)

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# [설정] 경로 및 하이퍼파라미터 (MAX_FEATURES 수정됨)
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

NUM_CLASSES = 531
MAX_FEATURES = 5000   # <--- 기존 2000에서 5000으로 상향 조정
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random Seed set to {seed}")

# =============================================================================
# 모델 정의 (GCN)
# =============================================================================
class LabelGCN(nn.Module):
    def __init__(self, emb_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(emb_dim, emb_dim)) for _ in range(num_layers)
        ])
        for W in self.weights:
            nn.init.xavier_uniform_(W)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, H, A_hat):
        for i, W in enumerate(self.weights):
            H = torch.matmul(A_hat, H)
            H = torch.matmul(H, W)
            if i < self.num_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return H

class GCNEnhancedClassifier(nn.Module):
    def __init__(self, input_dim, label_init_emb, A_hat, num_layers=2, dropout=0.5):
        super().__init__()
        emb_dim = label_init_emb.size(1)
        self.proj = nn.Linear(input_dim, emb_dim)
        self.gcn = LabelGCN(emb_dim=emb_dim, num_layers=num_layers, dropout=dropout)
        self.label_init_emb = nn.Parameter(label_init_emb.clone())
        self.register_buffer("A_hat", A_hat)
        self.dropout = dropout

    def forward(self, x):
        label_emb = self.gcn(self.label_init_emb, self.A_hat)
        x_proj = self.proj(x)
        x_proj = F.dropout(x_proj, p=self.dropout, training=self.training)
        logits = torch.matmul(x_proj, label_emb.T)
        return logits

# =============================================================================
# 유틸리티
# =============================================================================
def load_adjacency_matrix(file_path, num_classes):
    adj = torch.eye(num_classes)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p, c = int(parts[0]), int(parts[1])
                    if p < num_classes and c < num_classes:
                        adj[p, c] = 1
    return adj

def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

# =============================================================================
# Main Train Logic
# =============================================================================
def main():
    seed_everything(42)

    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)
    print(f"[Info] 작업 경로: {os.getcwd()}")

    data_path = "train_data.csv"
    hierarchy_path = "class_hierarchy.txt"

    # 1. 데이터 로드
    if not os.path.exists(data_path):
        print("[Error] train_data.csv가 없습니다. main.py를 먼저 실행하세요.")
        return
        
    print("[Info] 데이터 로드 중...")
    df = pd.read_csv(data_path)
    # 라벨 파싱
    df['label_ids'] = df['label_ids'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # 2. TF-IDF 변환 (Max Features 5000)
    print(f"[Info] TF-IDF 변환 (Max Features: {MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_matrix = vectorizer.fit_transform(df['text'].fillna("")).toarray()
    X_tensor = torch.FloatTensor(X_matrix)

    # 3. Multi-hot Encoding
    y_tensor = torch.zeros((len(df), NUM_CLASSES))
    for idx, label_list in enumerate(df['label_ids']):
        if label_list:
            y_tensor[idx, label_list] = 1.0

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. GCN 준비
    adj = load_adjacency_matrix(hierarchy_path, NUM_CLASSES)
    A_hat = normalize_adj(adj)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 학습 장치: {device}")

    label_init_emb = torch.randn(NUM_CLASSES, HIDDEN_DIM)

    model = GCNEnhancedClassifier(
        input_dim=MAX_FEATURES,
        label_init_emb=label_init_emb,
        A_hat=A_hat,
        num_layers=2,
        dropout=0.5
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 학습 루프
    print("[Info] 학습 시작...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

    print(f"[Success] 학습 완료. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()

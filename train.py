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
# [설정] 
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

NUM_CLASSES = 531
MAX_FEATURES = 5000
HIDDEN_DIM = 256        # 차원 확대
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15             # 1차 학습
SELF_TRAIN_EPOCHS = 15  # 2차 학습
CONFIDENCE_THR = 0.85   # 85% 이상 확신하는 것만 정답으로 채택

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
# 모델 (GCN) - 기존과 동일
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

def load_test_texts(path):
    texts = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    texts.append(parts[1])
    return texts

# =============================================================================
# Main (Self-Training Logic)
# =============================================================================
def main():
    seed_everything(42)

    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)

    data_path = "train_data.csv"
    test_path = "test_corpus.txt"
    hierarchy_path = "class_hierarchy.txt"

    # 1. 데이터 로드
    print("[Info] 데이터 로드 및 전처리...")
    df_train = pd.read_csv(data_path)
    df_train['label_ids'] = df_train['label_ids'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    test_texts = load_test_texts(test_path)
    
    # TF-IDF Fit (Train + Test 모두 사용하여 단어 사전 구축)
    print("[Info] TF-IDF 학습 (Train + Test)...")
    all_texts = df_train['text'].fillna("").tolist() + test_texts
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer.fit(all_texts)
    
    # Train 변환
    X_train = vectorizer.transform(df_train['text'].fillna("")).toarray()
    X_train_tensor = torch.FloatTensor(X_train)
    
    y_train = torch.zeros((len(df_train), NUM_CLASSES))
    for idx, label_list in enumerate(df_train['label_ids']):
        if label_list:
            y_train[idx, label_list] = 1.0

    train_dataset = TensorDataset(X_train_tensor, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 모델 초기화
    adj = load_adjacency_matrix(hierarchy_path, NUM_CLASSES)
    A_hat = normalize_adj(adj)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 초기화
    label_init_emb = torch.randn(NUM_CLASSES, HIDDEN_DIM)
    model = GCNEnhancedClassifier(
        input_dim=MAX_FEATURES, label_init_emb=label_init_emb, A_hat=A_hat, 
        num_layers=2, dropout=0.5
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------------------------------
    # Stage 1: 초기 학습 (Initial Training)
    # ----------------------------------------------------
    print("\n>>> [Stage 1] 초기 학습 시작 (Silver Labels) <<<")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

    # ----------------------------------------------------
    # Stage 2: Pseudo-Labeling (자가 라벨링)
    # ----------------------------------------------------
    print("\n>>> [Stage 2] Self-Training: 고신뢰도 샘플 추가 <<<")
    
    # Test 데이터 변환
    X_test = vectorizer.transform(test_texts).toarray()
    X_test_tensor = torch.FloatTensor(X_test)
    test_loader_unlabeled = DataLoader(TensorDataset(X_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    pseudo_samples = []
    pseudo_labels = []
    
    with torch.no_grad():
        for batch_idx, (bx,) in enumerate(test_loader_unlabeled):
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits)
            
            # 배치 내 각 샘플 확인
            for i in range(len(probs)):
                # 확률이 Threshold(0.85) 넘는 클래스 찾기
                high_conf_indices = (probs[i] > CONFIDENCE_THR).nonzero(as_tuple=False).squeeze(1)
                
                # 하나라도 확실한 게 있으면 훈련 데이터로 편입
                if len(high_conf_indices) > 0:
                    original_idx = batch_idx * BATCH_SIZE + i
                    pseudo_samples.append(X_test_tensor[original_idx])
                    
                    new_label = torch.zeros(NUM_CLASSES)
                    new_label[high_conf_indices.cpu()] = 1.0
                    pseudo_labels.append(new_label)

    print(f" -> 추가된 Pseudo-Label 데이터 수: {len(pseudo_samples)}개")

    if len(pseudo_samples) > 0:
        # 데이터 합치기
        X_pseudo = torch.stack(pseudo_samples)
        y_pseudo = torch.stack(pseudo_labels)
        
        X_combined = torch.cat([X_train_tensor, X_pseudo], dim=0)
        y_combined = torch.cat([y_train, y_pseudo], dim=0)
        
        new_dataset = TensorDataset(X_combined, y_combined)
        new_dataloader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # ----------------------------------------------------
        # Stage 3: 재학습 (Retraining)
        # ----------------------------------------------------
        print("\n>>> [Stage 3] 확장된 데이터로 재학습 (Retraining) <<<")
        for epoch in range(SELF_TRAIN_EPOCHS):
            model.train()
            total_loss = 0
            for bx, by in new_dataloader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Self-Epoch [{epoch+1}/{SELF_TRAIN_EPOCHS}] Loss: {total_loss/len(new_dataloader):.4f}")
    else:
        print(" -> 추가할 고신뢰도 샘플이 없습니다. 기존 모델을 저장합니다.")

    # 최종 저장
    torch.save(model.state_dict(), "best_model.pth")
    print(f"\n[Success] 모든 학습 완료. 모델 저장됨: best_model.pth")

if __name__ == "__main__":
    main()

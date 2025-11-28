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
# 1. 설정 및 시드 고정
# =============================================================================

# [사용자 설정] 데이터 파일이 있는 실제 경로 (Raw String 사용)
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"


# 하이퍼파라미터
NUM_CLASSES = 531
MAX_FEATURES = 2000   # TF-IDF 차원
HIDDEN_DIM = 128      # 라벨 임베딩 차원
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
# 2. 모델 클래스 정의 (제공된 코드)
# =============================================================================

class LabelGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network (GCN) encoder for label embeddings.
    """
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
    """
    Classifier that combines document representation and GCN-refined label embeddings.
    """
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
# 3. 유틸리티 함수 (그래프 구축 및 정규화)
# =============================================================================

def load_adjacency_matrix(file_path, num_classes):
    """
    class_hierarchy.txt를 읽어 531x531 인접 행렬 생성
    조건: 부모-자식=1, 대각선=1, 나머지=0
    """
    adj = torch.eye(num_classes) # 대각선 1로 초기화 (Self-loop)
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 파일 형식이 "ParentID ChildID" 라고 가정
                    p, c = int(parts[0]), int(parts[1])
                    if p < num_classes and c < num_classes:
                        adj[p, c] = 1
                        # GCN에서 정보 흐름을 원활하게 하기 위해 보통 무방향(Symmetric)으로 처리하거나
                        # 부모->자식 관계만 정의합니다. 여기서는 조건대로 관계만 1로 설정합니다.
    else:
        print("[Warning] 계층 파일이 없습니다. Identity Matrix를 사용합니다.")
    
    return adj

def normalize_adj(adj):
    """
    GCN 학습을 위한 인접 행렬 정규화 (Symmetric Normalization)
    D^{-0.5} * A * D^{-0.5}
    """
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

# =============================================================================
# 4. Main 실행 로직
# =============================================================================

def main():
    seed_everything(42)

    # 경로 설정
    if not os.path.exists(base_path):
        print(f"[Error] 경로를 찾을 수 없습니다: {base_path}")
        return
    os.chdir(base_path)
    print(f"[Info] Working Directory: {os.getcwd()}")

    data_path = "train_data.csv"
    hierarchy_path = "class_hierarchy.txt"

    # -------------------------------------------------------
    # 1. 데이터 로드 및 전처리
    # -------------------------------------------------------
    print("[Info] 데이터 로드 중...")
    df = pd.read_csv(data_path)
    
    # 문자열로 된 리스트 파싱 ("[1, 2]" -> [1, 2])
    print("[Info] Label Parsing (eval)...")
    df['label_ids'] = df['label_ids'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # TF-IDF 벡터화
    print(f"[Info] TF-IDF 변환 (Max Features: {MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_matrix = vectorizer.fit_transform(df['text'].fillna("")).toarray()
    X_tensor = torch.FloatTensor(X_matrix)

    # Multi-hot Encoding
    print("[Info] Multi-hot Label Encoding...")
    y_tensor = torch.zeros((len(df), NUM_CLASSES))
    for idx, label_list in enumerate(df['label_ids']):
        if label_list:
            # label_list에 있는 인덱스 위치를 1로 설정
            y_tensor[idx, label_list] = 1.0

    # Dataset & DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------------------------------------
    # 2. 그래프(Adjacency Matrix) 생성
    # -------------------------------------------------------
    print("[Info] Adjacency Matrix 생성 중...")
    adj = load_adjacency_matrix(hierarchy_path, NUM_CLASSES)
    
    # GCN 입력을 위한 정규화 (A_hat)
    A_hat = normalize_adj(adj)
    print(f"[Info] A_hat Shape: {A_hat.shape}")

    # -------------------------------------------------------
    # 3. 모델 초기화
    # -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # 라벨 초기 임베딩 (Random Initialization)
    # 실제로는 Word2Vec 등을 사용할 수도 있으나, 여기서는 랜덤값으로 시작하여 학습
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

    # -------------------------------------------------------
    # 4. 학습 루프
    # -------------------------------------------------------
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

        # Best Model 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            # print(" -> Best model saved.")

    print(f"[Success] 학습 완료. 최적 Loss: {best_loss:.4f}")
    print("저장된 모델: best_model.pth")

if __name__ == "__main__":
    main()

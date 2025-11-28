import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# =============================================================================
# 1. 설정 및 시드 고정
# =============================================================================

# [사용자 설정] 데이터 파일이 있는 실제 경로
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

# 하이퍼파라미터 (Train과 동일해야 함)
NUM_CLASSES = 531
MAX_FEATURES = 2000
HIDDEN_DIM = 128
BATCH_SIZE = 32

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
# 2. 모델 클래스 정의 (Train.py와 동일해야 가중치 로드 가능)
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
# 3. 유틸리티 함수 (그래프 로드 및 정규화)
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
# 4. Inference 실행
# =============================================================================

def main():
    seed_everything(42)

    # 경로 이동
    if not os.path.exists(base_path):
        print(f"[Error] 경로를 찾을 수 없습니다: {base_path}")
        return
    os.chdir(base_path)
    print(f"[Info] Working Directory: {os.getcwd()}")

    # 파일 경로
    train_file = "train_data.csv"
    test_file = "test_corpus.txt"
    hierarchy_file = "class_hierarchy.txt"
    model_path = "best_model.pth"
    output_file = "20252R0136DATA30400_final.csv"

    # 1. TF-IDF Vectorizer 준비
    # (학습 데이터로 Fit을 해야 feature space가 동일함)
    print("[Info] 훈련 데이터를 로드하여 Vectorizer Fit 수행 중...")
    if not os.path.exists(train_file):
        print("[Error] 훈련 데이터 파일이 없습니다.")
        return
    
    df_train = pd.read_csv(train_file)
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer.fit(df_train['text'].fillna("")) # FIT

    # 2. 테스트 데이터 로드 및 변환
    print(f"[Info] 테스트 데이터 로드: {test_file}")
    if not os.path.exists(test_file):
        print("[Error] 테스트 데이터 파일이 없습니다.")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f.readlines()]
    
    print(f" -> 총 {len(test_lines)}개의 테스트 문서")
    
    print("[Info] Vectorizer Transform 수행 중...")
    X_test_matrix = vectorizer.transform(test_lines).toarray()
    X_test_tensor = torch.FloatTensor(X_test_matrix)

    # DataLoader 생성 (배치 단위 추론을 위해)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 모델 구조 준비 및 가중치 로드
    print("[Info] 모델 초기화 및 가중치 로드...")
    
    # Adjacency Matrix 준비 (모델 초기화에 필요)
    adj = load_adjacency_matrix(hierarchy_file, NUM_CLASSES)
    A_hat = normalize_adj(adj)
    
    # 임의의 초기 임베딩 (Load State Dict 시 덮어씌워짐, Shape만 맞으면 됨)
    dummy_emb = torch.randn(NUM_CLASSES, HIDDEN_DIM)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GCNEnhancedClassifier(
        input_dim=MAX_FEATURES,
        label_init_emb=dummy_emb,
        A_hat=A_hat,
        num_layers=2,
        dropout=0.5
    ).to(device)

    if not os.path.exists(model_path):
        print(f"[Error] 모델 가중치 파일이 없습니다: {model_path}")
        return

    # 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 평가 모드 전환

    # 4. 추론 (Inference)
    print("[Info] 예측 수행 중 (Top-3)...")
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch[0].to(device)
            logits = model(inputs) # (Batch, Num_Classes)
            
            # 각 샘플별 상위 3개 클래스 인덱스 추출
            # topk_indices shape: (Batch, 3)
            _, topk_indices = torch.topk(logits, k=3, dim=1)
            
            # 리스트로 변환하여 저장
            for idx_list in topk_indices.cpu().numpy():
                # 정수형 인덱스를 문자열로 변환하여 공백으로 조인
                # 예: [1, 5, 10] -> "1 5 10"
                categories_str = " ".join(map(str, idx_list))
                all_predictions.append(categories_str)

    # 5. 결과 저장 (Submission Format)
    print("[Info] 제출 파일 생성 중...")
    submission = pd.DataFrame({
        'id': range(len(test_lines)), # 0부터 시작하는 ID
        'categories': all_predictions
    })

    submission.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] 완료! 파일 저장됨: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()

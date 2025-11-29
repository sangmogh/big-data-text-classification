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

# 하이퍼파라미터
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
# 2. 모델 클래스 정의 (Train.py와 동일)
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
# 3. 유틸리티 함수
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

def load_test_corpus(path):
    """
    test_corpus.txt 로드 (product_id \t text)
    Returns: list of pids, list of texts
    """
    pids = []
    texts = []
    
    print(f"[Info] 테스트 파일 로드 중: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pid, text = parts
                pids.append(pid)
                texts.append(text)
                
    print(f" -> 총 {len(pids)}개의 테스트 샘플 로드 완료.")
    return pids, texts

# =============================================================================
# 4. Inference Main Code
# =============================================================================

def main():
    seed_everything(42)

    # 1. 경로 설정
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

    # 2. Vectorizer 학습 (Train Data 기준)
    print("[Info] 훈련 데이터 로드 및 TF-IDF 학습 (Fit)...")
    if not os.path.exists(train_file):
        print("[Error] 훈련 데이터 파일이 없습니다. main.py를 먼저 실행하세요.")
        return
    
    df_train = pd.read_csv(train_file)
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer.fit(df_train['text'].fillna(""))

    # 3. 테스트 데이터 로드 및 변환
    if not os.path.exists(test_file):
        print("[Error] 테스트 코퍼스 파일이 없습니다.")
        return

    test_pids, test_texts = load_test_corpus(test_file) # PID와 Text 분리

    print("[Info] 테스트 텍스트 벡터화 (Transform)...")
    X_test_matrix = vectorizer.transform(test_texts).toarray()
    X_test_tensor = torch.FloatTensor(X_test_matrix)

    # DataLoader
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 모델 로드
    print("[Info] 모델 초기화 및 가중치 로드...")
    adj = load_adjacency_matrix(hierarchy_file, NUM_CLASSES)
    A_hat = normalize_adj(adj)
    
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
        print(f"[Error] 모델 가중치 파일({model_path})이 없습니다.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. 예측 수행
    print(f"[Info] 예측 수행 중 (Top-3)... Device: {device}")
    
    all_pred_strings = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch[0].to(device)
            logits = model(inputs)
            
            # 상위 3개 클래스 추출
            _, topk_indices = torch.topk(logits, k=3, dim=1)
            
            for idx_list in topk_indices.cpu().numpy():
                # 쉼표 구분 포맷 (예: "1,5,10")
                labels_str = ",".join(map(str, sorted(idx_list)))
                all_pred_strings.append(labels_str)

    # 6. 결과 저장 (수정됨: pid -> id)
    print("[Info] 제출 파일 생성 중...")
    
    submission = pd.DataFrame({
        'id': test_pids,           # 컬럼명을 'id'로 변경 (기존 pid 값 사용)
        'labels': all_pred_strings # 'labels'는 쉼표 구분 문자열
    })
    
    submission.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] 완료! 저장된 파일: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()

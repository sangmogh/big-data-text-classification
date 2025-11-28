import random
import numpy as np
import torch
import os

def seed_everything(seed=42):
    # 1. Python 내장 random 모듈
    random.seed(seed)
    
    # 2. OS 환경변수 (Hash 시드 고정 - 일부 라이브러리에서 필요)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    
    # 5. PyTorch 연산 결정론적 설정 (속도 저하 가능성 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 실행
seed_everything(42)

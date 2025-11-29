import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# =============================================================================
# [설정] 경로 (코랩용 ".")
# =============================================================================
base_path = "."
NUM_CLASSES = 531
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Seed set to {seed}")

def main():
    seed_everything(42)
    
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)

    # 1. 데이터 로드
    print("[Info] 데이터 로드 중...")
    if not os.path.exists("train_data.csv"):
        print("[Error] train_data.csv 없음. main.py 먼저 실행하세요.")
        return

    df = pd.read_csv("train_data.csv")
    df['label_ids'] = df['label_ids'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    # 라벨 없는 데이터 제거
    df = df[df['label_ids'].map(len) > 0]
    
    sentences = df.text.values
    labels = np.zeros((len(df), NUM_CLASSES))
    for idx, label_list in enumerate(df['label_ids']):
        if label_list:
            labels[idx, label_list] = 1.0

    # 2. 토크나이저
    print("[Info] BERT Tokenizer 로드 중...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # 3. 토큰화
    print("[Info] 텍스트 토큰화 중...")
    input_ids = []
    attention_masks = []

    for sent in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(
                            sent, 
                            add_special_tokens = True,
                            max_length = 128,
                            padding = 'max_length',  # [수정됨] 패딩 옵션
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.float)

    # 데이터셋
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

    # 4. 모델 로드
    print("[Info] BERT 모델 로드 중...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = NUM_CLASSES,
        problem_type = "multi_label_classification"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    # 5. 학습 루프
    for epoch_i in range(0, EPOCHS):
        print(f'\n======== Epoch {epoch_i + 1} / {EPOCHS} ========')
        model.train()
        
        # [수정] 변수명 통일 (total_loss -> total_train_loss)
        total_train_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            
            # [수정] 누적 변수명 일치
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        # [수정] 이제 에러 안 남
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")

    # 6. 저장
    print("\n[Info] 모델 저장 중...")
    model_save_path = "./bert_model_save"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("[Success] BERT 모델 저장 완료!")

if __name__ == "__main__":
    main()

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# =============================================================================
# [설정]
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"
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

def load_test_corpus(path):
    pids = []
    texts = []
    print(f"[Info] 테스트 파일 로드 중...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pids.append(parts[0])
                texts.append(parts[1])
    return pids, texts

def build_parent_map(hierarchy_path):
    child_to_parent = {}
    if os.path.exists(hierarchy_path):
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p, c = int(parts[0]), int(parts[1])
                    child_to_parent[c] = p
    return child_to_parent

def expand_labels(predicted_ids, child_to_parent):
    # 자식을 맞히면 부모도 정답에 추가
    expanded = set(predicted_ids)
    for pid in predicted_ids:
        curr = pid
        while curr in child_to_parent:
            parent = child_to_parent[curr]
            expanded.add(parent)
            curr = parent
            if curr in expanded and curr != pid: break
    return list(expanded)

def main():
    seed_everything(42)
    
    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)

    test_file = "test_corpus.txt"
    hierarchy_file = "class_hierarchy.txt"
    model_dir = "./bert_model_save"
    output_file = "20252R0136DATA30400_final_bert.csv"

    # 1. 테스트 데이터 로드
    test_pids, test_texts = load_test_corpus(test_file)

    # 2. 모델 로드
    print("[Info] 학습된 BERT 모델 로드 중...")
    if not os.path.exists(model_dir):
        print("[Error] 모델이 없습니다. train_bert.py 먼저 실행하세요.")
        return

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. 토큰화
    print("[Info] 테스트 데이터 토큰화...")
    input_ids = []
    attention_masks = []

    for sent in tqdm(test_texts, desc="Tokenizing"):
        encoded_dict = tokenizer.encode_plus(
                            sent, 
                            add_special_tokens = True, 
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)

    # 4. 추론
    print("[Info] 예측 수행 중...")
    child_to_parent = build_parent_map(hierarchy_file)
    all_pred_strings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits # (Batch, Num_Classes)
            
            # [핵심] 가장 확실한 1개(Top-1)만 뽑음 -> 오답 줄임
            _, topk_indices = torch.topk(logits, k=1, dim=1)

            for idx_list in topk_indices.cpu().numpy():
                # 계층 구조(부모) 추가
                expanded = expand_labels(idx_list, child_to_parent)
                
                # 최대 3개 제한 (과제 규칙 준수)
                if len(expanded) > 3: expanded = sorted(expanded)[:3]
                elif len(expanded) < 2: 
                    if 0 not in expanded: expanded.append(0)
                    if len(expanded) < 2: expanded.append(1)
                
                labels_str = ",".join(map(str, sorted(expanded)))
                all_pred_strings.append(labels_str)

    # 5. 저장
    submission = pd.DataFrame({'id': test_pids, 'labels': all_pred_strings})
    submission.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] 완료! 저장된 파일: {output_file}")

if __name__ == "__main__":
    main()

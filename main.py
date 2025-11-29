import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import string
from collections import defaultdict

# =============================================================================
# [설정] 경로 및 시드
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

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
# 데이터 로드 및 사전 구축 함수 (필터링 추가됨!)
# =============================================================================
def load_text_lines(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 파일을 찾을 수 없습니다: {filename}")
    print(f"[Info] '{filename}' 로딩 중...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def build_word_dictionary(filename):
    """
    키워드 파일을 읽어 {단어: {Class_ID...}} 딕셔너리를 구축합니다.
    [핵심 수정] 
    1. 2글자 이하 단어는 무시합니다. (예: 'on', 'at', 'go' 등 노이즈 제거)
    2. 너무 흔한 단어(불용어)도 무시합니다.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 키워드 파일을 찾을 수 없습니다: {filename}")

    print("[Info] 키워드 단어장(Dictionary) 구축 및 필터링 중...")
    
    word_to_class_ids = defaultdict(set)
    
    # 노이즈를 유발하는 흔한 단어 목록 (여기 있는 건 키워드여도 무시함)
    stop_words = {
        'the', 'and', 'for', 'with', 'one', 'new', 'top', 'set', 'pack', 
        'use', 'get', 'all', 'any', 'can', 'not', 'box', 'kit', 'pro', 'max'
    }

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for class_id, line in enumerate(lines):
            keywords = [k.strip().lower() for k in line.split(',') if k.strip()]
            
            for kw in keywords:
                # [필터링 1] 너무 짧은 단어 삭제 (2글자 이하)
                if len(kw) <= 2: 
                    continue
                
                # [필터링 2] 불용어 삭제
                if kw in stop_words:
                    continue

                # 통과한 단어만 등록
                word_to_class_ids[kw].add(class_id)
    
    print(f" -> 총 {len(word_to_class_ids)}개의 '유효한' 키워드가 등록되었습니다.")
    return word_to_class_ids

def labeling_logic_set(text, word_dict, translator):
    if not isinstance(text, str):
        return []
    
    # 1. 전처리: 소문자 변환 & 구두점 제거
    text_clean = text.lower().translate(translator)
    
    # 2. 토큰화: 공백 기준으로 잘라서 Set 생성
    tokens = set(text_clean.split())
    
    matched_ids = set()
    
    # 3. 조회: 키워드 사전에 있는 단어만 매칭
    for token in tokens:
        if token in word_dict:
            matched_ids.update(word_dict[token])
            
    return list(matched_ids)

# =============================================================================
# Main
# =============================================================================
def main():
    seed_everything(42)

    # 작업 경로 이동
    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)
    print(f"[Info] 작업 경로: {os.getcwd()}")

    corpus_file = 'train_corpus.txt'
    keyword_file = 'class_related_keywords.txt'
    output_file = 'train_data.csv'

    # 데이터 로드
    try:
        raw_lines = load_text_lines(corpus_file)
        word_dict = build_word_dictionary(keyword_file)
    except Exception as e:
        print(e)
        return

    # DataFrame 생성
    df = pd.DataFrame(raw_lines, columns=['text'])

    # 구두점 제거용 테이블
    translator = str.maketrans('', '', string.punctuation)

    # 라벨링 수행
    tqdm.pandas(desc="Filtered Labeling")
    print("[Info] 노이즈 필터링된 라벨링 수행 중...")
    
    df['label_ids'] = df['text'].progress_apply(
        lambda x: labeling_logic_set(x, word_dict, translator)
    )

    # 결과 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] '{output_file}' 저장 완료.")

if __name__ == "__main__":
    main()

import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import re

# =============================================================================
# [설정] 경로 및 시드
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"  # 기존 경로 유지

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
# 데이터 로드 및 정규식 컴파일 함수
# =============================================================================
def load_text_lines(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 파일을 찾을 수 없습니다: {filename}")
    print(f"[Info] '{filename}' 로딩 중...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def load_keywords_as_regex(filename):
    """
    각 클래스(줄)에 있는 키워드들을 '단어 경계(\\b)'가 포함된 정규표현식으로 컴파일합니다.
    예: keywords=["case", "cover"] -> Regex: r'\b(?:case|cover)\b'
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 키워드 파일을 찾을 수 없습니다: {filename}")

    regex_map = {}
    print("[Info] 키워드 파일 로드 및 Regex(단어 경계) 컴파일 중...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # 1. 키워드 분리 및 전처리
            keywords = [k.strip() for k in line.split(',') if k.strip()]
            
            if keywords:
                # 2. 각 키워드에 대해 이스케이프 처리 (특수문자 오작동 방지)
                # 단어 경계(\b)를 앞뒤로 붙여서 정확한 단어 매칭 유도
                escaped_keywords = [re.escape(k) for k in keywords]
                
                # 3. 하나의 거대한 OR 패턴 생성: \b(?:kw1|kw2|kw3)\b
                # (?:...)는 Non-capturing group
                pattern_str = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
                
                # 4. 컴파일 (IGNORECASE: 대소문자 무시)
                regex_map[idx] = re.compile(pattern_str, re.IGNORECASE)
    
    print(f" -> 총 {len(regex_map)}개의 클래스에 대해 Regex 패턴 생성 완료.")
    return regex_map

def labeling_logic_regex(text, regex_map):
    """
    컴파일된 Regex를 사용하여 텍스트 내 정확한 단어 매칭 확인
    """
    if not isinstance(text, str):
        return []
    
    matched_ids = []
    # 모든 클래스 패턴에 대해 검사 (속도는 조금 느려질 수 있으나 정확도 우선)
    for class_id, pattern in regex_map.items():
        if pattern.search(text):
            matched_ids.append(class_id)
            
    return matched_ids

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
        regex_map = load_keywords_as_regex(keyword_file)
    except Exception as e:
        print(e)
        return

    # DataFrame 생성
    df = pd.DataFrame(raw_lines, columns=['text'])

    # 라벨링 수행
    tqdm.pandas(desc="Regex Labeling")
    print("[Info] 정밀 라벨링(Regex \\b) 수행 중...")
    df['label_ids'] = df['text'].progress_apply(lambda x: labeling_logic_regex(x, regex_map))

    # 결과 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] '{output_file}' 저장 완료.")

if __name__ == "__main__":
    main()

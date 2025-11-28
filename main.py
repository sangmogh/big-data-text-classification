import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# [설정 구역] 사용자 환경에 맞게 아래 경로를 수정하세요.
# Windows 경로 에러 방지를 위해 반드시 r"..." (Raw String) 형식을 유지하세요.
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"


def seed_everything(seed=42):
    """
    재현성을 위해 모든 시드를 고정합니다.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] 모든 Random Seed가 {seed}로 고정되었습니다.")


def load_text_lines(filename):
    """
    Python 내장 open()을 사용하여 텍스트를 리스트로 읽어옵니다.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 파일을 찾을 수 없습니다: {filename}")

    print(f"[Info] '{filename}' 로딩 중...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    print(f" -> 총 {len(lines)}개의 라인을 읽었습니다.")
    return lines


def build_inverted_index(filename):
    """
    [핵심 최적화 1] Inverted Index 생성
    
    Returns:
        1. inverted_index: { '단어': {class_id_1, class_id_2, ...} }
        2. class_keywords: { class_id: ['original keyword', ...] }
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[Error] 키워드 파일을 찾을 수 없습니다: {filename}")

    print("[Info] Inverted Index(역색인) 구축 중...")
    
    inverted_index = defaultdict(set) # 단어 -> 관련 Class ID 집합
    class_keywords = {}               # Class ID -> 전체 키워드 리스트 (검증용)

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for class_id, line in enumerate(lines):
            # 키워드 파싱
            keywords = [k.strip().lower() for k in line.split(',') if k.strip()]
            class_keywords[class_id] = keywords
            
            # 역색인 구성
            for kw in keywords:
                # 키워드를 공백 기준으로 쪼개서, 구성 단어(Token) 각각에 Class ID를 매핑
                # 예: "deep learning" -> 'deep'과 'learning' 각각에 class_id 추가
                tokens = kw.split()
                for token in tokens:
                    inverted_index[token].add(class_id)
    
    print(f" -> 역색인 구축 완료. 총 {len(inverted_index)}개의 고유 단어가 인덱싱되었습니다.")
    return inverted_index, class_keywords


def labeling_logic_inverted(text, inverted_index, class_keywords_map):
    """
    [핵심 최적화 2] 역색인을 이용한 2단계 라벨링
    1단계: 텍스트에 등장한 단어들을 이용해 '후보 Class'를 추립니다.
    2단계: 후보 Class의 키워드들만 실제로 텍스트에 있는지 정밀 검사(String Match)합니다.
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    
    # 1. 텍스트 토큰화 (단순 공백 분리)
    # set으로 만들어 중복 제거 및 빠른 조회
    text_tokens = set(text_lower.split())
    
    # 2. 후보 Class ID 추리기 (Candidates Filtering)
    candidate_class_ids = set()
    for token in text_tokens:
        # 리뷰에 있는 단어가 역색인에 있다면, 관련 Class ID들을 후보군에 추가
        if token in inverted_index:
            candidate_class_ids.update(inverted_index[token])
            
    # 3. 후보군 정밀 검사 (Verification)
    matched_ids = []
    
    # 전체 클래스가 아니라, 후보로 추려진 소수의 클래스만 검사
    for class_id in candidate_class_ids:
        keywords = class_keywords_map[class_id]
        # 해당 클래스의 키워드 중 하나라도 텍스트 원문에 포함되어 있는지 확인
        for kw in keywords:
            if kw in text_lower:
                matched_ids.append(class_id)
                break # 하나라도 찾으면 해당 클래스는 확정, 다음 클래스로
                
    return matched_ids


def main():
    # 1. 시드 고정
    seed_everything(42)

    # 2. 작업 경로 변경
    print(f"[Info] 작업 경로를 설정합니다: {base_path}")
    try:
        os.chdir(base_path)
    except FileNotFoundError:
        print(f"[Error] 경로를 찾을 수 없습니다: {base_path}")
        return

    # 파일명 정의
    corpus_file = 'train_corpus.txt'
    keyword_file = 'class_related_keywords.txt'
    output_file = 'train_data.csv'

    # 3. 데이터 로드 및 인덱스 구축
    try:
        # 코퍼스 로드
        raw_lines = load_text_lines(corpus_file)
        
        # 키워드 파일 로드 및 Inverted Index 생성
        inverted_index, class_keywords_map = build_inverted_index(keyword_file)
        
    except Exception as e:
        print(e)
        return

    # 4. DataFrame 변환
    df = pd.DataFrame(raw_lines, columns=['text'])

    # 5. Inverted Index 라벨링 수행
    tqdm.pandas(desc="Index-Based Labeling") 
    
    print("[Info] 역색인 기반 고속 라벨링 시작...")
    # apply에 필요한 인자들(inverted_index, class_keywords_map)을 lambda로 전달
    df['label_ids'] = df['text'].progress_apply(
        lambda x: labeling_logic_inverted(x, inverted_index, class_keywords_map)
    )

    # 6. 결과 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] 완료! 저장 경로: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()

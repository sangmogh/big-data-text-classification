import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# =============================================================================
# [설정] 경로 및 시드
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch는 여기선 안 쓰지만 일관성을 위해 유지
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Info] Random Seed set to {seed}")

# =============================================================================
# 로직 함수
# =============================================================================

def load_class_documents(base_path):
    """
    각 클래스의 키워드들을 공백으로 이어붙여 '하나의 문서'로 만듭니다.
    Return: [ "keyword1 keyword2 ...", "keyword3 ...", ... ] (총 531개 문자열)
    """
    keyword_file = os.path.join(base_path, 'class_related_keywords.txt')
    if not os.path.exists(keyword_file):
        raise FileNotFoundError(f"키워드 파일 없음: {keyword_file}")
    
    class_docs = []
    print("[Info] 클래스 키워드 로드 및 문서화 중...")
    
    with open(keyword_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 쉼표를 공백으로 치환하여 하나의 문자열로 결합
            # 예: "apple, banana" -> "apple banana"
            keywords = line.strip().replace(',', ' ')
            class_docs.append(keywords)
            
    print(f" -> 총 {len(class_docs)}개의 클래스 문서 생성 완료.")
    return class_docs

def load_test_corpus(base_path):
    """
    test_corpus.txt 로드 (pid \t text)
    """
    test_file = os.path.join(base_path, 'test_corpus.txt')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"테스트 파일 없음: {test_file}")
    
    pids = []
    texts = []
    print("[Info] 테스트 데이터 로드 중...")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pids.append(parts[0])
                texts.append(parts[1])
                
    print(f" -> 총 {len(texts)}개의 테스트 문서 로드 완료.")
    return pids, texts

def main():
    seed_everything(42)
    
    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)

    # 1. 데이터 로드
    class_docs = load_class_documents(base_path) # List of strings (Classes)
    test_pids, test_texts = load_test_corpus(base_path) # List of strings (Reviews)

    # 2. TF-IDF 벡터화
    print("[Info] TF-IDF 벡터화 수행 중...")
    
    # [설정 설명]
    # analyzer='word': 단어 단위 분석
    # stop_words='english': a, the, is 같은 무의미한 단어 자동 제거
    # token_pattern: 3글자 이상의 단어만 취급 (r'(?u)\b\w\w\w+\b') -> 노이즈 제거 효과
    # sublinear_tf=True: 단어 빈도가 무한히 커질 때 가중치를 로그 스케일로 줄임 (TF smoothing)
    vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        token_pattern=r'(?u)\b\w\w\w+\b', 
        sublinear_tf=True,
        max_features=10000 # 충분히 크게 잡음
    )
    
    # 클래스 문서들을 기준으로 단어장(Vocabulary)을 학습(Fit)합니다.
    # 즉, "클래스 키워드에 없는 단어"는 무시됩니다. (이게 핵심!)
    X_classes = vectorizer.fit_transform(class_docs)
    
    # 테스트 데이터는 Transform만 수행
    X_tests = vectorizer.transform(test_texts)
    
    print(f" -> Class Vector Shape: {X_classes.shape}")
    print(f" -> Test Vector Shape: {X_tests.shape}")

    # 3. 코사인 유사도 계산 및 Top-3 추출
    print("[Info] 코사인 유사도 계산 및 Top-3 클래스 선정 중...")
    
    # 메모리 효율을 위해 배치(Batch) 처리 할 수도 있지만, 
    # 텍스트 데이터가 수만 건 정도면 한 번에 계산 가능합니다.
    # (Reviews x Vocab) @ (Vocab x Classes).T = (Reviews x Classes)
    similarity_matrix = cosine_similarity(X_tests, X_classes)
    
    all_pred_strings = []
    
    # 진행상황 표시
    for i in tqdm(range(similarity_matrix.shape[0]), desc="Inference"):
        scores = similarity_matrix[i]
        
        # 점수가 높은 순으로 정렬하여 인덱스 가져오기 (argsort는 오름차순이므로 뒤에서부터 자름)
        # 상위 3개 추출
        top3_indices = scores.argsort()[-3:][::-1]
        
        # 만약 상위 1등 점수가 0점이라면? (겹치는 단어가 아예 없는 경우)
        # -> 일단 0,1,2번 클래스가 찍히겠지만, 
        #    이런 경우는 어쩔 수 없이 베이스라인(랜덤 등)과 같음.
        
        # 결과 포맷팅 (오름차순 정렬 후 쉼표로 결합)
        # 예: class 10, class 5, class 100 -> "5,10,100"
        labels_str = ",".join(map(str, sorted(top3_indices)))
        all_pred_strings.append(labels_str)

    # 4. 결과 저장
    output_file = "20252R0136DATA30400_final.csv"
    submission = pd.DataFrame({
        'id': test_pids,
        'labels': all_pred_strings
    })
    
    submission.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] 완료! 파일 저장됨: {os.path.abspath(output_file)}")
    
    # 샘플 출력
    print("\n[Sample Output]")
    print(submission.head())

if __name__ == "__main__":
    main()

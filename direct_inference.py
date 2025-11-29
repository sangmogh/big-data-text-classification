import pandas as pd
from tqdm import tqdm
import os
import re
from collections import defaultdict, Counter

# =============================================================================
# [설정] 경로
# =============================================================================
base_path = r"C:\Users\wangm\Documents\Final project\20252R0136DATA30400"

# =============================================================================
# 로직
# =============================================================================
def load_test_corpus(path):
    pids = []
    texts = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    pids.append(parts[0])
                    texts.append(parts[1])
    return pids, texts

def build_keyword_map(filename):
    """키워드 맵 생성 (필터링 강화)"""
    word_to_class = defaultdict(set)
    # 노이즈 불용어
    stop_words = {'the', 'and', 'for', 'with', 'one', 'new', 'top', 'set', 'pack', 
                  'use', 'get', 'all', 'any', 'can', 'not', 'box', 'kit', 'pro', 'max',
                  'product', 'item', 'amazon', 'good', 'great', 'best', 'love', 'like'}

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for class_id, line in enumerate(lines):
                keywords = [k.strip().lower() for k in line.split(',') if k.strip()]
                for kw in keywords:
                    if len(kw) <= 3 or kw in stop_words: continue # 3글자 이하 버림
                    # 단어장에 등록
                    word_to_class[kw].add(class_id)
    return word_to_class

def build_parent_map(hierarchy_path):
    """자식 -> 부모 매핑"""
    child_to_parent = {}
    if os.path.exists(hierarchy_path):
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p, c = int(parts[0]), int(parts[1])
                    child_to_parent[c] = p
    return child_to_parent

def get_ancestors(class_id, child_to_parent):
    """특정 클래스의 모든 조상을 찾음"""
    ancestors = set()
    curr = class_id
    while curr in child_to_parent:
        parent = child_to_parent[curr]
        ancestors.add(parent)
        curr = parent
        if curr in ancestors: break # cycle 방지
    return ancestors

def predict_rule_based(text, word_to_class, child_to_parent):
    text_lower = text.lower()
    
    # 1. 텍스트 내 단어 찾기 (Regex로 단어 추출)
    tokens = re.findall(r'[a-z]{3,}', text_lower)
    
    # 등장한 클래스 빈도수 체크
    class_counter = Counter()
    
    for token in tokens:
        if token in word_to_class:
            for cid in word_to_class[token]:
                class_counter[cid] += 1
    
    # 2. 후보 선정
    if not class_counter:
        return [0, 1] 

    # 가장 유력한 Leaf Class 1~2개를 선정
    top_candidates = [c for c, _ in class_counter.most_common(2)]
    
    final_set = set(top_candidates)
    
    # 3. 계층 구조 확장 (Parent Propagation)
    for cid in top_candidates:
        ancestors = get_ancestors(cid, child_to_parent)
        final_set.update(ancestors)
        
    # 4. 개수 맞추기 (2~3개)
    # [수정 완료] 변수명을 result로 통일했습니다.
    result = sorted(list(final_set))
    
    if len(result) > 3:
        result = result[:3] # 앞에서부터 3개
    elif len(result) < 2:
        if 0 not in result: result.insert(0, 0)
        if len(result) < 2: result.append(1)

    return result

# =============================================================================
# Main
# =============================================================================
def main():
    if not os.path.exists(base_path):
        print(f"[Error] 경로 없음: {base_path}")
        return
    os.chdir(base_path)

    test_file = "test_corpus.txt"
    keyword_file = "class_related_keywords.txt"
    hierarchy_file = "class_hierarchy.txt"
    output_file = "20252R0136DATA30400_final_direct.csv"

    # 로드
    pids, texts = load_test_corpus(test_file)
    word_to_class = build_keyword_map(keyword_file)
    child_to_parent = build_parent_map(hierarchy_file)

    print(f"[Info] Rule-based 예측 시작 ({len(texts)}개)...")
    
    results = []
    for text in tqdm(texts):
        labels = predict_rule_based(text, word_to_class, child_to_parent)
        labels_str = ",".join(map(str, labels))
        results.append(labels_str)

    # 저장 (pid -> id)
    df = pd.DataFrame({'id': pids, 'labels': results})
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[Success] '{output_file}' 저장 완료.")

if __name__ == "__main__":
    main()

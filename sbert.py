import os
import random
import numpy as np
import torch
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Union, Optional, Set, Any
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

# ==========================================
# 1. Reproducibility Setup
# ==========================================
def set_seed(seed: int = 42) -> None:
    """
    재현성을 위해 random, numpy, torch의 시드를 고정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Setup] All seeds fixed to {seed}")

# ==========================================
# 2. Class Definitions
# ==========================================

class TaxonomyLoader:
    """
    계층적 라벨 구조(Taxonomy)를 로드하고 그래프(DAG) 정보를 관리하는 클래스
    """
    def __init__(self, classes_path: str, hierarchy_path: str, keywords_path: Optional[str] = None) -> None:
        self.classes_path = classes_path
        self.hierarchy_path = hierarchy_path
        self.keywords_path = keywords_path
        
        self.label_map: Dict[str, int] = {} # Name -> ID
        self.id_map: Dict[int, str] = {}    # ID -> Name
        self.graph: nx.DiGraph = nx.DiGraph()
        
        self._load_data()

    def _load_data(self) -> None:
        self._load_classes(self.classes_path)
        if self.label_map:
            self._load_hierarchy(self.hierarchy_path)
        if self.keywords_path and self.label_map:
            self._load_keywords(self.keywords_path)

    def _load_classes(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[Warning] Classes file not found: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # ID<tab>Name 형식 파싱
                        idx, name = int(parts[0]), parts[1]
                        self.label_map[name] = idx
                        self.id_map[idx] = name
                        self.graph.add_node(idx, name=name)
            print(f"[Loader] Loaded {len(self.label_map)} classes.")
        except Exception as e:
            print(f"[Error] Loading classes: {e}")

    def _load_hierarchy(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            edges_count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        # ParentID<tab>ChildID
                        parent_id, child_id = int(parts[0]), int(parts[1])
                        if parent_id in self.id_map and child_id in self.id_map:
                            self.graph.add_edge(parent_id, child_id)
                            edges_count += 1
            print(f"[Loader] Loaded hierarchy with {edges_count} edges.")
        except Exception as e:
            print(f"[Error] Loading hierarchy: {e}")

    def _load_keywords(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Name: kw1, kw2...
                    if ':' in line:
                        name_part, keywords_part = line.strip().split(':', 1)
                        label_name = name_part.strip()
                        if label_name in self.label_map:
                            node_id = self.label_map[label_name]
                            keywords = [k.strip() for k in keywords_part.split(',') if k.strip()]
                            self.graph.nodes[node_id]['keywords'] = keywords
            print(f"[Loader] Loaded keywords.")
        except Exception as e:
            print(f"[Error] Loading keywords: {e}")

    def get_graph(self) -> nx.DiGraph:
        return self.graph
    
    def get_id_map(self) -> Dict[int, str]:
        return self.id_map


class ReviewDataset(Dataset):
    """
    텍스트 데이터를 로드 및 관리하는 클래스.
    입력 파일 형식: ID <tab> Product_Name <tab> Review_Text 처리를 강화함.
    """
    def __init__(self, data_samples: List[Dict[str, Union[int, str]]]) -> None:
        self.samples = data_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    @staticmethod
    def load_corpus(file_path: str) -> List[Dict[str, Union[int, str]]]:
        """
        파일을 읽어 ID와 Review Text를 추출합니다.
        오류가 발생한 라인은 건너뛰고 경고를 출력합니다.
        """
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            return []
        
        data = []
        parse_errors = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 1. 탭으로 우선 분리 시도
                    parts = line.split('\t')
                    
                    # 2. 탭 분리가 잘 안되고 공백 분리 로직이 필요한 경우 (예비책)
                    # 데이터 명세가 'ID(int) Product(str) Review(str)' 이므로 최소 3개 부분 예상
                    # 하지만 리뷰 텍스트 내에 탭이 없을 거라 가정하고 마지막 요소를 리뷰로 취급
                    if len(parts) >= 3:
                        row_id = int(parts[0]) # 첫 번째는 ID
                        text = parts[-1]       # 마지막은 Review Text
                    else:
                        # 형식이 맞지 않는 경우 공백으로 시도하거나 건너뜀
                        # 여기서는 안전하게 건너뛰고 카운트
                        parse_errors += 1
                        continue

                    data.append({'id': row_id, 'text': text})
                    
                except ValueError:
                    # ID가 숫자가 아닌 경우 등
                    parse_errors += 1
                    continue

        if parse_errors > 0:
            print(f"[Data] Warning: Skipped {parse_errors} lines due to parsing errors.")
        
        print(f"[Data] Successfully loaded {len(data)} samples from {os.path.basename(file_path)}.")
        return data


class SemanticClassifier(torch.nn.Module):
    """
    SBERT 임베딩 및 True Path Rule 기반의 DAG 후처리를 포함한 분류기
    """
    def __init__(self, model_name: str, taxonomy_loader: TaxonomyLoader, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        print(f"[Model] Loading SBERT model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        
        self.taxonomy = taxonomy_loader
        self.graph = taxonomy_loader.get_graph()
        self.id_map = taxonomy_loader.get_id_map()
        
        # 라벨 임베딩 사전 계산
        self.label_embeddings = self._precompute_label_embeddings()

    def _precompute_label_embeddings(self) -> torch.Tensor:
        sorted_ids = sorted(self.id_map.keys())
        texts_to_encode = []
        for idx in sorted_ids:
            name = self.id_map[idx]
            keywords = self.graph.nodes[idx].get('keywords', [])
            # 텍스트 = "이름: 키워드1, 키워드2..."
            text = f"{name}: {', '.join(keywords)}" if keywords else name
            texts_to_encode.append(text)
            
        embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.to(self.device)

    def forward(self, input_texts: List[str], batch_size: int = 32) -> torch.Tensor:
        input_embeddings = self.model.encode(
            input_texts, 
            batch_size=batch_size, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            device=self.device
        )
        # Cosine Similarity: [Batch, Num_Labels]
        scores = util.cos_sim(input_embeddings, self.label_embeddings)
        return scores

    def predict(self, texts: List[str], top_k_candidates: int = 15, batch_size: int = 32) -> List[List[str]]:
        print(f"[Predict] Starting prediction for {len(texts)} samples...")
        # 1. 유사도 계산
        scores_tensor = self.forward(texts, batch_size=batch_size)
        
        final_predictions = []
        # 2. DAG Constraint 적용
        for i in tqdm(range(scores_tensor.shape[0]), desc="Applying DAG Constraints"):
            scores = scores_tensor[i]
            pred_ids = self._apply_true_path_rule(scores, top_k_candidates)
            pred_labels = [self.id_map[idx] for idx in pred_ids]
            final_predictions.append(pred_labels)
            
        return final_predictions

    def _apply_true_path_rule(self, scores: torch.Tensor, top_k: int) -> List[int]:
        """
        True Path Rule 적용 로직:
        1. 점수가 높은 Anchor부터 선택.
        2. Anchor의 모든 조상(Ancestors)을 강제 포함.
        3. 결과 집합이 3개 미만이면 다음 후보 Anchor 탐색.
        4. 3개 초과 시 'Leaf 노드' 중 '점수'가 낮은 순으로 제거 (Truncation).
        """
        # 점수 상위 k개 후보 인덱스 추출
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
        
        selected_indices: Set[int] = set()
        
        # --- Expansion Phase ---
        for anchor_idx in top_indices:
            # 이미 포함된 노드라면 스킵 (상위 노드로 인해 포함되었을 수 있음)
            if anchor_idx in selected_indices:
                continue
            
            # True Path Rule: Anchor + Ancestors
            ancestors = nx.ancestors(self.graph, anchor_idx)
            current_path = ancestors | {anchor_idx}
            
            # 합집합 업데이트
            selected_indices.update(current_path)
            
            # 최소 3개 이상 채워지면 일단 멈추고 Truncation 단계로 이동
            if len(selected_indices) >= 3:
                break
        
        selected_list = list(selected_indices)
        
        # --- Truncation Phase (최대 3개 유지) ---
        # 조건: 3개를 초과할 경우, DAG 구조상 제거해도 안전한(자식이 없는) Leaf 노드부터 제거
        while len(selected_list) > 3:
            # 현재 선택된 노드들로만 구성된 서브그래프
            subgraph = self.graph.subgraph(selected_list)
            
            # Out-Degree가 0인 노드 == 현재 집합 내에서의 Leaf
            leaves = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
            
            if not leaves: 
                # (이론상 DAG에서는 발생 안 함) 안전장치: 점수 꼴찌 제거
                leaves = selected_list
            
            # Leaf 중 원본 유사도 점수가 가장 낮은 노드(victim) 선정
            victim = min(leaves, key=lambda idx: scores[idx].item())
            
            selected_list.remove(victim)
            
        # 최종 반환: 점수 높은 순으로 정렬 (가독성 위함)
        selected_list.sort(key=lambda idx: scores[idx].item(), reverse=True)
        return selected_list

# ==========================================
# 3. Main Pipeline Execution
# ==========================================
def main():
    # 1. 시드 고정
    set_seed(42)
    
    # 2. 경로 설정
    base_path = "." 
    config = {
        'classes': os.path.join(base_path, 'classes.txt'),
        'hierarchy': os.path.join(base_path, 'class_hierarchy.txt'),
        'keywords': os.path.join(base_path, 'class_related_keywords.txt'),
        'test_corpus': os.path.join(base_path, 'test_corpus.txt'),
        'output': os.path.join(base_path, 'submission.csv'),
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"[System] Running on device: {config['device']}")

    # 3. Taxonomy Loader 초기화
    print("-" * 30)
    taxonomy = TaxonomyLoader(
        classes_path=config['classes'], 
        hierarchy_path=config['hierarchy'], 
        keywords_path=config['keywords']
    )

    # 4. Semantic Classifier 초기화
    print("-" * 30)
    classifier = SemanticClassifier(
        model_name=config['model_name'], 
        taxonomy_loader=taxonomy, 
        device=config['device']
    )

    # 5. 데이터 로드 (파싱 로직 강화됨)
    print("-" * 30)
    data_samples = ReviewDataset.load_corpus(config['test_corpus'])
    if not data_samples:
        print("[Error] No valid test data found. Exiting.")
        return

    # 데이터셋 객체 생성
    test_dataset = ReviewDataset(data_samples)
    
    # 입력 텍스트 리스트 추출
    test_texts = [sample['text'] for sample in test_dataset]
    test_ids = [sample['id'] for sample in test_dataset]

    # 6. 예측 실행
    print("-" * 30)
    predictions = classifier.predict(
        test_texts, 
        batch_size=config['batch_size']
    )

    # 7. 결과 저장 (id, labels 형식 준수)
    print("-" * 30)
    results = []
    for row_id, preds in zip(test_ids, predictions):
        # 공백으로 구분된 라벨 문자열 생성
        label_str = " ".join(preds)
        results.append({'id': row_id, 'labels': label_str})
    
    df_submission = pd.DataFrame(results)
    
    # 컬럼 순서 명시적 지정 (id, labels) 및 저장
    df_submission = df_submission[['id', 'labels']]
    df_submission.to_csv(config['output'], index=False, encoding='utf-8')
    
    print(f"[Output] Submission saved to {config['output']}")
    print("Done.")

if __name__ == "__main__":
    main()

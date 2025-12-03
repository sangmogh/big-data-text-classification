import os
import random
import numpy as np
import torch
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Union, Optional, Set
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
    
    def get_label_map(self) -> Dict[str, int]:
        return self.label_map

    def get_id_map(self) -> Dict[int, str]:
        return self.id_map


class ReviewDataset(Dataset):
    """
    텍스트 데이터를 로드 및 관리하는 클래스
    """
    def __init__(self, texts: List[str], labels: Optional[List[List[str]]] = None) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        item = {'text': text}
        if self.labels:
            item['labels'] = self.labels[idx]
        return item
    
    @staticmethod
    def load_corpus(file_path: str) -> List[str]:
        """텍스트 코퍼스 파일을 줄 단위로 읽어 리스트로 반환"""
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]


class SemanticClassifier(torch.nn.Module):
    """
    SBERT 임베딩 및 DAG 후처리를 포함한 분류기
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
        # Cosine Similarity
        scores = util.cos_sim(input_embeddings, self.label_embeddings)
        return scores

    def predict(self, texts: List[str], top_k_candidates: int = 15, batch_size: int = 32) -> List[List[str]]:
        print(f"[Predict] Starting prediction for {len(texts)} samples...")
        scores_tensor = self.forward(texts, batch_size=batch_size)
        
        final_predictions = []
        for i in tqdm(range(scores_tensor.shape[0]), desc="Applying DAG Constraints"):
            scores = scores_tensor[i]
            pred_ids = self._apply_dag_constraint(scores, top_k_candidates)
            pred_labels = [self.id_map[idx] for idx in pred_ids]
            final_predictions.append(pred_labels)
            
        return final_predictions

    def _apply_dag_constraint(self, scores: torch.Tensor, top_k: int) -> List[int]:
        # 점수 상위 k개 후보 인덱스
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
        selected_indices: Set[int] = set()
        
        # 1. Anchor 탐색 및 조상 추가
        for anchor_idx in top_indices:
            if anchor_idx in selected_indices:
                continue
            
            # 자신 포함 모든 조상 가져오기
            ancestors = nx.ancestors(self.graph, anchor_idx)
            chain = ancestors | {anchor_idx}
            selected_indices.update(chain)
            
            # 최소 3개 이상이면 1차 중단 (단, 너무 많아지면 아래에서 자름)
            if len(selected_indices) >= 3:
                break
        
        selected_list = list(selected_indices)
        
        # 2. 개수 제어 (3개 초과 시 Leaf 노드부터 Pruning)
        # 'Leaf'이면서 '유사도 점수'가 낮은 순서대로 제거
        while len(selected_list) > 3:
            subgraph = self.graph.subgraph(selected_list)
            leaves = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
            
            if not leaves: # 사이클 등 예외 상황
                leaves = selected_list
                
            # Leaf 중 점수가 가장 낮은 것 찾기
            victim = min(leaves, key=lambda idx: scores[idx].item())
            selected_list.remove(victim)
            
        # 점수 높은 순 정렬 반환
        selected_list.sort(key=lambda idx: scores[idx].item(), reverse=True)
        return selected_list

# ==========================================
# 3. Main Pipeline Execution
# ==========================================
def main():
    # 1. 시드 고정
    set_seed(42)
    
    # 2. 경로 설정 (업로드된 파일명 기준)
    base_path = "."  # 현재 경로
    config = {
        'classes': os.path.join(base_path, 'classes.txt'),
        'hierarchy': os.path.join(base_path, 'class_hierarchy.txt'),
        'keywords': os.path.join(base_path, 'class_related_keywords.txt'),
        'test_corpus': os.path.join(base_path, 'test_corpus.txt'),
        'output': os.path.join(base_path, 'submission.csv'),
        'model_name': 'all-MiniLM-L6-v2', # 가볍고 성능 좋은 SBERT 모델
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

    # 5. 데이터 로드 (ReviewDataset 활용)
    print("-" * 30)
    test_texts = ReviewDataset.load_corpus(config['test_corpus'])
    if not test_texts:
        print("[Error] No test data found. Exiting.")
        return

    # 데이터셋 객체 생성 (배치 처리를 위해 필요 시 DataLoader로 확장 가능)
    # 여기서는 predict 메서드에 리스트를 직접 전달
    test_dataset = ReviewDataset(texts=test_texts)
    print(f"[Data] Loaded {len(test_dataset)} test samples.")

    # 6. 예측 실행
    print("-" * 30)
    predictions = classifier.predict(
        test_dataset.texts, 
        batch_size=config['batch_size']
    )

    # 7. 결과 저장
    print("-" * 30)
    # Submission 포맷: text_index, label_string
    # label_string은 공백으로 구분된 라벨 이름들로 가정
    results = []
    for i, preds in enumerate(predictions):
        label_str = " ".join(preds)
        results.append({'id': i, 'labels': label_str})
    
    df_submission = pd.DataFrame(results)
    df_submission.to_csv(config['output'], index=False)
    print(f"[Output] Submission saved to {config['output']}")
    print("Done.")

if __name__ == "__main__":
    main()

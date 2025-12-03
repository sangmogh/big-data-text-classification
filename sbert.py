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
    
    def get_id_map(self) -> Dict[int, str]:
        return self.id_map


class ReviewDataset(Dataset):
    """
    텍스트 데이터를 로드 및 관리하는 클래스.
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
        파일을 읽어 '첫 번째 공백'을 기준으로 ID와 나머지 텍스트를 분리합니다.
        ID는 정수로 변환하며, 변환 실패 시 해당 라인은 건너뜁니다.
        """
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            return []
        
        data = []
        parse_errors = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 첫 번째 공백(Space/Tab) 기준으로 최대 1번만 분리 (ID / Text)
                parts = line.split(maxsplit=1)
                
                if len(parts) < 2:
                    parse_errors += 1
                    continue
                    
                row_id_str, text_content = parts[0], parts[1]
                
                try:
                    row_id = int(row_id_str)
                    data.append({'id': row_id, 'text': text_content})
                except ValueError:
                    parse_errors += 1
                    continue

        if parse_errors > 0:
            print(f"[Data] Warning: Skipped {parse_errors} lines due to parsing/format errors.")
        
        print(f"[Data] Successfully loaded {len(data)} samples from {os.path.basename(file_path)}.")
        return data


class SemanticClassifier(torch.nn.Module):
    """
    SBERT 임베딩 및 True Path Rule 기반의 DAG 후처리를 포함한 분류기.
    """
    def __init__(self, model_name: str, taxonomy_loader: TaxonomyLoader, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        print(f"[Model] Loading SBERT model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        
        self.taxonomy = taxonomy_loader
        self.graph = taxonomy_loader.get_graph()
        self.id_map = taxonomy_loader.get_id_map()
        
        # [중요] 임베딩 텐서의 인덱스와 실제 Taxonomy ID를 매핑하기 위한 정렬된 리스트
        self.sorted_label_ids = sorted(self.id_map.keys())
        
        # 라벨 임베딩 사전 계산
        self.label_embeddings = self._precompute_label_embeddings()

    def _precompute_label_embeddings(self) -> torch.Tensor:
        """
        sorted_label_ids 순서대로 임베딩을 생성합니다.
        """
        texts_to_encode = []
        for idx in self.sorted_label_ids:
            name = self.id_map[idx]
            keywords = self.graph.nodes[idx].get('keywords', [])
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

    def predict(self, texts: List[str], top_k_candidates: int = 15, batch_size: int = 32) -> List[List[int]]:
        """
        예측 결과로 클래스 이름 대신 실제 클래스 ID(int)의 리스트를 반환합니다.
        """
        print(f"[Predict] Starting prediction for {len(texts)} samples...")
        scores_tensor = self.forward(texts, batch_size=batch_size)
        
        final_predictions_ids = []
        for i in tqdm(range(scores_tensor.shape[0]), desc="Applying DAG Constraints"):
            scores = scores_tensor[i]
            # ID 리스트 반환
            pred_ids = self._apply_true_path_rule(scores, top_k_candidates)
            final_predictions_ids.append(pred_ids)
            
        return final_predictions_ids

    def _apply_true_path_rule(self, scores: torch.Tensor, top_k: int) -> List[int]:
        """
        True Path Rule + Truncation 적용
        반환값: 실제 Taxonomy ID들의 리스트
        """
        # 1. Tensor Index(0~N) 기준 Top K 추출
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
        
        selected_actual_ids: Set[int] = set()
        
        # 2. Tensor Index -> Actual ID 변환 및 Expansion 수행
        for tensor_idx in top_indices:
            # 매핑 정보를 이용해 실제 ID 획득
            actual_id = self.sorted_label_ids[tensor_idx]
            
            if actual_id in selected_actual_ids:
                continue
            
            # networkx 그래프는 실제 ID를 노드로 가짐
            ancestors = nx.ancestors(self.graph, actual_id)
            current_path = ancestors | {actual_id}
            selected_actual_ids.update(current_path)
            
            # 최소 3개 이상 채워지면 중단 (이후 Truncation으로 조절)
            if len(selected_actual_ids) >= 3:
                break
        
        selected_list = list(selected_actual_ids)
        
        # 3. Truncation: 3개 초과 시 Leaf 제거
        # 유사도 점수를 조회하기 위해 Actual ID -> Tensor Index 역매핑 필요
        # 하지만 sorted_label_ids는 정렬되어 있으므로, binary search나 dict를 쓸 수 있으나
        # 여기서는 scores에 접근하기 위해 id_to_tensor_idx 맵을 잠시 생성하거나, 
        # 그냥 sorted_label_ids.index()를 쓰면 느릴 수 있음.
        # 성능을 위해 dict 미리 생성 권장. 여기서는 가독성을 위해 간단히 dict comprehension 사용.
        id_to_tensor_idx = {uid: idx for idx, uid in enumerate(self.sorted_label_ids)}

        while len(selected_list) > 3:
            subgraph = self.graph.subgraph(selected_list)
            # 현재 선택된 집합 내에서의 Leaf (Out-Degree 0)
            leaves = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
            
            if not leaves: 
                leaves = selected_list
                
            # Leaf 중 원본 유사도 점수가 가장 낮은 것 제거
            # scores[tensor_index]를 참조해야 함
            victim = min(leaves, key=lambda uid: scores[id_to_tensor_idx[uid]].item())
            selected_list.remove(victim)
            
        # 4. 최종 정렬 (점수 높은 순)
        selected_list.sort(key=lambda uid: scores[id_to_tensor_idx[uid]].item(), reverse=True)
        return selected_list

# ==========================================
# 3. Main Pipeline Execution
# ==========================================
def main():
    set_seed(42)
    
    # 경로 설정
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

    # 로더 초기화
    taxonomy = TaxonomyLoader(config['classes'], config['hierarchy'], config['keywords'])
    
    # 모델 초기화
    classifier = SemanticClassifier(config['model_name'], taxonomy, config['device'])

    # 데이터 로드
    print("-" * 30)
    data_samples = ReviewDataset.load_corpus(config['test_corpus'])
    
    if not data_samples:
        print("[Error] No valid test data found after parsing. Exiting.")
        return

    test_texts = [s['text'] for s in data_samples]
    test_ids = [s['id'] for s in data_samples]

    # 예측 (ID 리스트 반환)
    print("-" * 30)
    predictions_ids = classifier.predict(test_texts, batch_size=config['batch_size'])

    # 저장 (ID를 쉼표로 연결)
    print("-" * 30)
    results = []
    for row_id, pred_ids in zip(test_ids, predictions_ids):
        # [수정] ID(int) 리스트를 쉼표로 구분된 문자열로 변환 (예: "10,25,30")
        label_str = ",".join(map(str, pred_ids))
        results.append({'id': row_id, 'labels': label_str})
    
    df_submission = pd.DataFrame(results)
    df_submission = df_submission[['id', 'labels']]
    df_submission.to_csv(config['output'], index=False, encoding='utf-8')
    
    print(f"[Output] Submission saved to {config['output']}")
    print("Done.")

if __name__ == "__main__":
    main()

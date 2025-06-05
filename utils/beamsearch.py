import torch
import random
from collections import defaultdict
from utils.node_aspect_mapper import NodeAspectMapper

# KGAT의 학습된 임베딩을 활용해서 u-i를 연결하는 유의미한 경로를 탐색함.
# 이 경로들을 LLM에 넣을 수 있게 텍스트로 직렬화하고
# 아이템 정보와 user history도 함꼐 제공해서
# llm 기반 추천 설명을 위한 재료를 제공함
class CollaborativeBeamSearchWithAspect: 
    '''BeamSearch를 하는데 노드 당 다음 단계에 연결된 rel을 10개로 제한.하고 그 중 top beam_num개만 남기는 방식
    '''
    def __init__(self, data, model,
                 user_aspect_dict=None,          # {"감정선":0.32, …}
                 node_mapper:NodeAspectMapper=None,
                 lambda_aspect:float=0.6,
                 max_cand_per_node : int = 10, # Beam search 할 때 한 노드 당 탐색 폭 10개가 최대로 설정
                 args = None):
        """초기화 함수.
        
        Args:
            data: 데이터셋 객체. 학습 데이터와 매핑 정보를 포함한다.
            model: 학습된 모델 객체. 임베딩 정보를 포함한다.
        """
        self.data = data
        self.model = model
        self.all_embeddings = self.model.entity_user_embed.weight
        self.entity_id_maps = [self.data.entity_id2org, self.data.user_id2org]
        self.relation_id_maps = [self.data.relation_id2org] # [{2: 'item_has_subject_as_attribute:',3: 'item_has_director_as_attribute:',
        self.user_aspect_dict = user_aspect_dict or {}   # {"감정선":0.32, ...}
        self.mapper = node_mapper or NodeAspectMapper()
        self.lambda_aspect    = lambda_aspect
        self.max_cand_per_node = max_cand_per_node # Beam search 할 때 한 노드 당 탐색 폭 10개가 최대로 설정
        self.args = args
        
    def _compute_cosine_scores(self, candidate_embeddings, reference_embedding):
        """주어진 참조 임베딩과 후보 임베딩 간의 코사인 유사도 점수를 계산합니다.
        
        Args:
            candidate_embeddings: 후보 임베딩 텐서.
            reference_embedding: 참조 임베딩 텐서.
        
        Returns:
            Tensor: 후보들과 참조 임베딩 간의 코사인 유사도 점수.
        """
        """주어진 참조 임베딩과 후보 임베딩 간의 코사인 유사도 점수 계산"""
        """Compute cosine similarity scores between the reference embedding and candidate embeddings."""
        return torch.nn.functional.cosine_similarity(candidate_embeddings, reference_embedding, dim=1)
    
    # 현재 노드에서 다음으로 확장 가능한 후보 노드들을 수집
    # 각 후보는 user 및 item과의 코사인 유사도 기반 점수를 계산
    def _get_candidates(self, user_id, item_id, next_id, visited, only_attribute,prefix_score=0):
        """return 후보 노드 리스트, 후보 relation 리스트, 후보노드들 cosine&aspect 반영 점수 리스트
            다음 확장 노드를 위한 후보와 그들의 관계 및 점수를 가져옵니다. 
        
        Args:
            user_id: 탐색 중인 사용자 노드 ID
            item_id: 최종적으로 연결하려는 타깃 아이템 노드 ID
            next_id: 지금 확장하려는 현재(head) 노드 ID
            visited: 방문한 노드의 집합.이미 경로에 들어간 노드 집합 → 중복 방문 방지
            only_attribute: True면 속성(edge)만 확장(예: director, genre), False면 모든 edge 사용
            prefix_score: 	이전 hop까지 누적된 경로 점수
        
        Returns:
            tuple: 후보 노드들, 관계들, 평균 점수들.
        """
        if only_attribute: # True면 속성(edge)만 확장.
            candidates = [element[0] for element in self.data.train_kg_dict[next_id] if element[0] not in visited and element[1] not in [0,1]]
            relations = [element[1] for element in self.data.train_kg_dict[next_id] if element[0] not in visited and element[1] not in [0,1]]

        else:
            candidates = [element[0] for element in self.data.train_kg_dict[next_id] if element[0] not in visited]
            relations = [element[1] for element in self.data.train_kg_dict[next_id] if element[0] not in visited] # [ 2, 1,4,,....]
        
        candidate_embeddings = self.all_embeddings[candidates]
        user_embedding = self.all_embeddings[user_id].unsqueeze(0)
        item_embedding = self.all_embeddings[item_id].unsqueeze(0)
        
        # 사용자와 아이템에 대한 코사인 유사도 점수 계산
        user_scores = self._compute_cosine_scores(candidate_embeddings, user_embedding)
        item_scores = self._compute_cosine_scores(candidate_embeddings, item_embedding)
        
        # 평균 코사인 유사도 점수
        # average_scores = (torch.mean(torch.stack([user_scores, item_scores]), dim=0) + prefix_score).tolist()
        # base score (cosine)
        base_scores = torch.mean(torch.stack([user_scores, item_scores]), dim=0)

        # aspect 가중치
        aspect_w = torch.zeros_like(base_scores)
        if self.user_aspect_dict: # {감정선 : 0.32, ...}
            asp_list = list(self.user_aspect_dict.keys()) # ["감정선", ...]
            
            # candidates에서 5개씩 llm에 넣어서 관련된 aspect 뽑아내기
            # 5개씩 묶어서 llm에 넣어야함
            node_names = [self.data.entity_id2org[c] for c in candidates]
            
            batched_nodes = [node_names[i:i+5] for i in range(0, len(node_names), 5)]

            aspect_scores = {}
            for batch in batched_nodes: # [a1, a2, a3,a4,a5]가 batch
                result_dict = self.mapper.judge_nodes(self.args, batch, asp_list)
                for node, related_aspects in result_dict.items():
                    max_score = max([self.user_aspect_dict[a] for a in related_aspects]) if related_aspects else 0
                    aspect_scores[node] = max_score

            # aspect_w 설정
            for idx, cand_id in enumerate(candidates):
                node_name = self.data.entity_id2org[cand_id]
                aspect_w[idx] = aspect_scores.get(node_name, 0)
                

        # ③ 최종 score
        # average_scores = (base_scores + self.lambda_aspect * aspect_w + prefix_score).tolist()
        # base_socres 범위 : [-1, 1] 
        # aspect_w 범위 : [0, 1]
        # 후보 노드들에 대해 각각 average_scores 계산한거임.
        average_scores = (base_scores + self.lambda_aspect * aspect_w + prefix_score).tolist()
        return candidates, relations, average_scores        

    def _sort_beam_nodes(self, beam_nodes, num_beams, fill=False):
        """
        후보 노드 리스트(beam_nodes)를 점수를 기준으로 정렬해서 상위 num_beams 개만 남기는 함수
        부족하면 fill=True일 때 랜덤하게 채움
        빔 노드들을 점수에 따라 정렬하고, 필요시 샘플링을 통해 채웁니다.
        
        Args:
            beam_nodes: 빔 탐색 중의 노드 리스트. 현재 hop에서 확장된 후보 노드들: [head, relation, tail, score] 형태
            num_beams: 유지할 빔의 수.
            fill: 빔의 수가 부족할 경우 채울지 여부.
        
        Returns:
            list: 정렬되고 필요시 채워진 빔 노드 리스트.
        """
        if fill and len(beam_nodes) < num_beams:
            if len(beam_nodes) > 0:
                random.seed(self.args.seed)
                sampled_nodes = random.choices(beam_nodes, k=num_beams - len(beam_nodes))
                beam_nodes.extend(sampled_nodes)
            else:
                return beam_nodes
        return sorted(beam_nodes, key=lambda x: x[3], reverse=True)[:num_beams]
    
    def _sort_beam_paths(self, beam_paths, num_beams, fill=False):
        """빔 경로들을 점수에 따라 정렬하고, 필요시 샘플링을 통해 채웁니다.
        
        Args:
            beam_paths: 빔 탐색 중의 경로 리스트.
            num_beams: 유지할 빔의 수.
            fill: 빔의 수가 부족할 경우 채울지 여부.
        
        Returns:
            list: 정렬되고 필요시 채워진 빔 경로 리스트.
        """
        if fill and len(beam_paths) < num_beams:
            if len(beam_paths) > 0:
                random.seed(self.args.seed)
                sampled_paths = random.choices(beam_paths, k=num_beams - len(beam_paths) )
                beam_paths.extend(sampled_paths)
            else:
                return beam_paths
        return sorted(beam_paths, key=lambda x: x[-1][3], reverse=True)[:num_beams]
    
    # 방문 순서만 다른 중복 경로 제거
    def remove_duplicate_paths(self, paths): # 방문 순서만 다르고 방문한 노드가 같은 경로 제거
        """방문 순서만 다른 중복 경로를 제거합니다.
        
        Args:
            paths: 경로 리스트.
        
        Returns:
            list: 중복을 제거한 경로 리스트.
        """
        unique_paths = []  # 중복되지 않은 경로를 저장할 리스트
        visited_nodes_sets = set()  # 각 경로의 방문 노드 집합
        
        for path in paths:
            current_set = tuple(sorted({triplet[i] for triplet in path for i in [0, 2]}))
            if current_set not in visited_nodes_sets:
                unique_paths.append(path)
                visited_nodes_sets.add(current_set)
        return unique_paths

    # user 와 item을 연결하는 경로 탐색(Beam Search 기반)
    # 빔서치 방식으로 최대 num_hops까지 경로 탐색
    # 경로는 [(h, r, t, score), ...] 형태의 triplet 리스트
    # 마지막 hop에서는 item_id로 끝나는 경로만 유효 경로로 간주
    def search(self, user_id, item_id, only_attribute=False, remove_duplicate=True, num_beams=100, num_hops=5):
        """빔 탐색을 수행하여 경로를 찾습니다.
        
        Args:
            user_id: 사용자 ID. (int)
            item_id: 아이템 ID. (int)
            only_attribute: 속성 관계만을 대상으로 할지의 여부.
            remove_duplicate: 중복 경로 제거 여부.
            num_beams: 빔의 수. 지금 노드에서 갈 수 있는 후보 노드에서 점수 기준으로 상위 num_beams개 살림
            num_hops: 탐색할 홉의 수.
        
        Returns:
            tuple: 유효한 경로와 모든 경로.
        """
        paths = []
        for hop in range(1, num_hops + 1):
            if hop == 1:
                visited = {user_id, item_id}
                candidates, relations, prefix_scores = self._get_candidates(user_id, item_id, user_id, visited, only_attribute=False, prefix_score=0)
                print(f"[DEBUG] {hop}")
                expanded_nodes = [[user_id, relation_, tail_, prefix_score_] for tail_, relation_, prefix_score_ in zip(candidates, relations, prefix_scores)]
                sorted_expanded_nodes = self._sort_beam_nodes(expanded_nodes, num_beams, fill=False)
                paths = [[tuple(triplet_info),] for triplet_info in sorted_expanded_nodes]
                    # paths = [                                                
                    #     [(user_id, rel₁, tail₁, score₁)],
                    #     [(user_id, rel₂, tail₂, score₂)],
                    #     ...
                    # ]paths는 "user로부터 시작하는 1-hop 후보 경로들
                    # 예시
                    # [
                    #     [(23, 1, 87, 0.76)],     # user 23 → relation 1 → node 87
                    #     [(23, 4, 91, 0.72)],     # user 23 → relation 4 → node 91
                    #     [(23, 7, 55, 0.65)],     # ...
                    #     ...
                    # ]
            elif hop < num_hops:
                candi = []
                
                print(f"[DEBUG] Hop {hop}: {len(paths)} paths to expand")
                for idx, path in enumerate(paths):
                    _, _, tail, prefix_score = path[-1]
                    
                    visited = set([triplet_info[2] for path_ in paths for triplet_info in path_])
                    visited.add(user_id)
                    
                    candidates, relations, prefix_scores = self._get_candidates(user_id, item_id, tail, visited, only_attribute=only_attribute, prefix_score=prefix_score)
                    print(f"[DEBUG] Hop {hop}, path {idx}: {len(candidates)} candidates to expand")

                    
                    next_head = tail
                    expanded_nodes = [[next_head, relation_, tail_, prefix_score_] for tail_, relation_, prefix_score_ in zip(candidates, relations, prefix_scores)]
                    
                    # 하나의 빔에서 뻗어 나온 가지들 중 num_beams개 가져오기
                    fill = True if hop == num_hops - 1 else False # 마지막에서 두번째 hop에서는 부족할 경우 beam만큼 채워줌.
                    sorted_expanded_nodes = self._sort_beam_nodes(expanded_nodes, num_beams, fill=fill)
                    
                    candi.extend([path + [tuple(triplet_info)] for triplet_info in sorted_expanded_nodes])
                if remove_duplicate:
                    candi = self.remove_duplicate_paths(candi)
                paths = self._sort_beam_paths(candi, num_beams)

                print(f"[DEBUG] Hop {hop}: Total expanded paths = {len(candi)}")
                
                print(f"[INFO] {len(paths)} paths found")
                print(f"[INFO] paths example : {paths[0]}")
        
            elif hop == num_hops:
                connected = 0
                for idx, path in enumerate(paths):
                    visited = set([triplet_info[2] for triplet_info in path])
                    visited.add(user_id)
                    _, _, tail, prefix_score = path[-1]
                    for element in self.data.train_kg_dict[tail]:
                        next_node, next_relation = element
                        
                        if next_node == item_id:
                            connected += 1
                            final_path = path + [tuple([tail, next_relation, item_id, None])]
                            break
                        else:
                            final_path = path + [tuple([None] * 4)]
                    paths[idx] = final_path
                
                print(f"[DEBUG] Final hop: {connected} paths connect to item {item_id}")
                print(f"[INFO] {len(paths)} paths found")
                print(f"[INFO] paths example : {paths[0]}")
                    
        valid_paths = [path for path in paths if path[-1][0] != None]
        valid_paths = self.remove_duplicate_paths(valid_paths)
        print(f"[INFO] {valid_paths}")
        return valid_paths, paths   
    
    
    # 내부 ID를 실제 이름으로 변환
    def entity_id2original_name(self, node_id):
        for id_map in self.entity_id_maps:
            if node_id in id_map:
                return id_map[node_id]
        raise Exception(f"Entity {node_id} does not Exist!")
    
    def relation_id2original_name(self, relation_id):
        for id_map in self.relation_id_maps:
            if relation_id in id_map:
                return id_map[relation_id]
        raise Exception(f"Relation {relation_id} does not Exist!")
    
    def triplet2original_name(self, triplet:tuple):
        head, relation, tail, prefix_score = triplet
        return self.entity_id2original_name(int(head)), self.relation_id2original_name(int(relation)), \
               self.entity_id2original_name(int(tail)), prefix_score
    
    # KG 경로를 자연어 문장처럼 직렬화(LLM에 넣기 위해)
    # 추출된 triplet 경로들을 llm이 읽을 수 있도록 문자열로 바꿔줌
    # 점수도 함께 반환 -> 나중에 top-K 선택에 사용가능
    def path2linearlize(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                    if to_original_name else triplet
                str_path += (f'{head} -> {relation} -> ')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_path += (f'{tail}')
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    def path2triplet(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                    if to_original_name else triplet
                str_path += (f'{list((head, relation, tail))}\n')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    def path2organize(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                    if to_original_name else triplet
                str_path += (f'{list((head, relation, tail))}\n')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    # 아이템의 메타데이터 정보 생성(감독, 장르 등)
    # 아이템의 속성 정보를 텍스트로 출력. LLM 프롬프트에 들어갈 "감독은 누구고, 장르는 무엇" 같은 문장 생성
    def item_information(self, iid, max_relations = 5):
        result = []
        meta_information = defaultdict(list)
        for eid, rid in self.data.train_kg_dict[iid]:
            if rid != 0 and eid < self.data.n_entities:
                meta_information[rid].append(self.data.entity_id2org[eid])
        for rid, entity_list in list(meta_information.items())[:max_relations]:
            relation_name = self.data.relation_id2org[rid].replace('item_has_','').replace('_as_attribute:', '')
            entity_list_str = ' '.join(entity_list)
            result.append(f'The {relation_name} of the movie is/are {entity_list_str}.')
        return '\n'.join(result)
    
    # 유저가 본 아이템들의 메타데이터 요약 리스트 생성
    # user의 속성 정보를 텍스트로 출력해서 LLM에 프롬프트로 들어갈 문장 생성
    def user_history(self, uid, max_items,max_lines=10):
        result = []
        prefered_item_list = self.data.train_kg_dict[uid]
        if len(prefered_item_list) > max_items:
            prefered_item_list = random.sample(prefered_item_list, k=max_items)
        for idx, (prefered_iid, rid) in enumerate(prefered_item_list):
            item_info = self.item_information(prefered_iid, max_relations=3)  # 관계 
            result.append(f'{idx + 1}. {item_info}')
        result_text = '\n'.join(result)
        lines = result_text.split('\n')[:max_lines]
        return '\n'.join(lines)
    
    
# KGAT의 학습된 임베딩을 활용해서 u-i를 연결하는 유의미한 경로를 탐색함.
# 이 경로들을 LLM에 넣을 수 있게 텍스트로 직렬화하고
# 아이템 정보와 user history도 함꼐 제공해서
# llm 기반 추천 설명을 위한 재료를 제공함

class CollaborativeBeamSearch:
    def __init__(self, data, model):
        """초기화 함수.
        
        Args:
            data: 데이터셋 객체. 학습 데이터와 매핑 정보를 포함한다.
            model: 학습된 모델 객체. 임베딩 정보를 포함한다.
        """
        self.data = data
        self.model = model
        self.all_embeddings = self.model.entity_user_embed.weight
        self.entity_id_maps = [self.data.entity_id2org, self.data.user_id2org]
        self.relation_id_maps = [self.data.relation_id2org]

    def _compute_cosine_scores(self, candidate_embeddings, reference_embedding):
        """주어진 참조 임베딩과 후보 임베딩 간의 코사인 유사도 점수를 계산합니다.
        
        Args:
            candidate_embeddings: 후보 임베딩 텐서.
            reference_embedding: 참조 임베딩 텐서.
        
        Returns:
            Tensor: 후보들과 참조 임베딩 간의 코사인 유사도 점수.
        """
        """주어진 참조 임베딩과 후보 임베딩 간의 코사인 유사도 점수 계산"""
        """Compute cosine similarity scores between the reference embedding and candidate embeddings."""
        return torch.nn.functional.cosine_similarity(candidate_embeddings, reference_embedding, dim=1)

    def _get_candidates(self, user_id, item_id, next_id, visited, only_attribute, prefix_score=0):
        """다음 확장 노드를 위한 후보와 그들의 관계 및 점수를 가져옵니다.
        
        Args:
            user_id: 사용자 ID.
            item_id: 아이템 ID.
            next_id: 다음 확장할 노드의 ID.
            visited: 방문한 노드의 집합.
            only_attribute: 속성 관계만을 대상으로 할지의 여부.
            prefix_score: 현재까지의 점수.
        
        Returns:
            tuple: 후보 노드들, 관계들, 평균 점수들.
        """
        if only_attribute:
            candidates = [element[0] for element in self.data.train_kg_dict[next_id] if element[0] not in visited and element[1] not in [0, 1]]
            relations = [element[1] for element in self.data.train_kg_dict[next_id] if element[0] not in visited and element[1] not in [0, 1]]
        else:
            candidates = [element[0] for element in self.data.train_kg_dict[next_id] if element[0] not in visited]
            relations = [element[1] for element in self.data.train_kg_dict[next_id] if element[0] not in visited]
        
        candidate_embeddings = self.all_embeddings[candidates]
        user_embedding = self.all_embeddings[user_id].unsqueeze(0)
        item_embedding = self.all_embeddings[item_id].unsqueeze(0)
        
        # 사용자와 아이템에 대한 코사인 유사도 점수 계산
        user_scores = self._compute_cosine_scores(candidate_embeddings, user_embedding)
        item_scores = self._compute_cosine_scores(candidate_embeddings, item_embedding)
        
        # 평균 코사인 유사도 점수
        average_scores = (torch.mean(torch.stack([user_scores, item_scores]), dim=0) + prefix_score).tolist()
        
        return candidates, relations, average_scores

    def _sort_beam_nodes(self, beam_nodes, num_beams, fill=False):
        """빔 노드들을 점수에 따라 정렬하고, 필요시 샘플링을 통해 채웁니다.
        
        Args:
            beam_nodes: 빔 탐색 중의 노드 리스트.
            num_beams: 유지할 빔의 수.
            fill: 빔의 수가 부족할 경우 채울지 여부.
        
        Returns:
            list: 정렬되고 필요시 채워진 빔 노드 리스트.
        """
        if fill and len(beam_nodes) < num_beams:
            if len(beam_nodes) > 0:
                sampled_nodes = random.choices(beam_nodes, k=num_beams - len(beam_nodes))
                beam_nodes.extend(sampled_nodes)
            else:
                return beam_nodes
        return sorted(beam_nodes, key=lambda x: x[3], reverse=True)[:num_beams]
    
    def _sort_beam_paths(self, beam_paths, num_beams, fill=False):
        """빔 경로들을 점수에 따라 정렬하고, 필요시 샘플링을 통해 채웁니다.
        
        Args:
            beam_paths: 빔 탐색 중의 경로 리스트.
            num_beams: 유지할 빔의 수.
            fill: 빔의 수가 부족할 경우 채울지 여부.
        
        Returns:
            list: 정렬되고 필요시 채워진 빔 경로 리스트.
        """
        if fill and len(beam_paths) < num_beams:
            if len(beam_paths) > 0:
                sampled_paths = random.choices(beam_paths, k=num_beams - len(beam_paths))
                beam_paths.extend(sampled_paths)
            else:
                return beam_paths
        return sorted(beam_paths, key=lambda x: x[-1][3], reverse=True)[:num_beams]
    
    def remove_duplicate_paths(self, paths): # 방문 순서만 다르고 방문한 노드가 같은 경로 제거
        """방문 순서만 다른 중복 경로를 제거합니다.
        
        Args:
            paths: 경로 리스트.
        
        Returns:
            list: 중복을 제거한 경로 리스트.
        """
        unique_paths = []  # 중복되지 않은 경로를 저장할 리스트
        visited_nodes_sets = set()  # 각 경로의 방문 노드 집합
        
        for path in paths:
            current_set = tuple(sorted({triplet[i] for triplet in path for i in [0, 2]}))
            if current_set not in visited_nodes_sets:
                unique_paths.append(path)
                visited_nodes_sets.add(current_set)
        return unique_paths

    def search(self, user_id, item_id, only_attribute=False, remove_duplicate=True, num_beams=100, num_hops=5):
        """빔 탐색을 수행하여 경로를 찾습니다.
        
        Args:
            user_id: 사용자 ID. (int)
            item_id: 아이템 ID. (int)
            only_attribute: 속성 관계만을 대상으로 할지의 여부.
            remove_duplicate: 중복 경로 제거 여부.
            num_beams: 빔의 수.
            num_hops: 탐색할 홉의 수.
        
        Returns:
            tuple: 유효한 경로와 모든 경로.
        """
        paths = []
        for hop in range(1, num_hops + 1):
            if hop == 1:
                visited = {user_id, item_id}
                candidates, relations, prefix_scores = self._get_candidates(user_id, item_id, user_id, visited, only_attribute=False, prefix_score=0)
                expanded_nodes = [[user_id, relation_, tail_, prefix_score_] for tail_, relation_, prefix_score_ in zip(candidates, relations, prefix_scores)]
                sorted_expanded_nodes = self._sort_beam_nodes(expanded_nodes, num_beams, fill=False)
                paths = [[tuple(triplet_info),] for triplet_info in sorted_expanded_nodes]
            elif hop < num_hops:
                candi = []
                
                for idx, path in enumerate(paths):
                    _, _, tail, prefix_score = path[-1]
                    
                    visited = set([triplet_info[2] for path_ in paths for triplet_info in path_])
                    visited.add(user_id)
                    
                    candidates, relations, prefix_scores = self._get_candidates(user_id, item_id, tail, visited, only_attribute=only_attribute, prefix_score=prefix_score)
                    
                    next_head = tail
                    expanded_nodes = [[next_head, relation_, tail_, prefix_score_] for tail_, relation_, prefix_score_ in zip(candidates, relations, prefix_scores)]
                    
                    # 하나의 빔에서 뻗어 나온 가지들 중 num_beams개 가져오기
                    fill = True if hop == num_hops - 1 else False # 마지막에서 두번째 hop에서는 부족할 경우 beam만큼 채워줌.
                    sorted_expanded_nodes = self._sort_beam_nodes(expanded_nodes, num_beams, fill=fill)
                    
                    candi.extend([path + [tuple(triplet_info)] for triplet_info in sorted_expanded_nodes])
                if remove_duplicate:
                    candi = self.remove_duplicate_paths(candi)
                paths = self._sort_beam_paths(candi, num_beams)
                
            elif hop == num_hops:
                for idx, path in enumerate(paths):
                    visited = set([triplet_info[2] for triplet_info in path])
                    visited.add(user_id)
                    _, _, tail, prefix_score = path[-1]
                    for element in self.data.train_kg_dict[tail]:
                        next_node, next_relation = element
                        if next_node == item_id:
                            final_path = path + [tuple([tail, next_relation, item_id, None])]
                            break
                    else:
                        final_path = path + [tuple([None] * 4)]
                    paths[idx] = final_path
        valid_paths = [path for path in paths if path[-1][0] != None]
        valid_paths = self.remove_duplicate_paths(valid_paths)
        return valid_paths, paths
    
    def entity_id2original_name(self, node_id):
        for id_map in self.entity_id_maps:
            if node_id in id_map:
                return id_map[node_id]
        raise Exception(f"Entity {node_id} does not Exist!")
    
    def relation_id2original_name(self, relation_id):
        for id_map in self.relation_id_maps:
            if relation_id in id_map:
                return id_map[relation_id]
        raise Exception(f"Relation {relation_id} does not Exist!")
    
    def triplet2original_name(self, triplet:tuple):
        head, relation, tail, prefix_score = triplet
        return self.entity_id2original_name(int(head)), self.relation_id2original_name(int(relation)), \
               self.entity_id2original_name(int(tail)), prefix_score
    
    def path2linearlize(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                     if to_original_name else triplet
                str_path += (f'{head} -> {relation} -> ')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_path += (f'{tail}')
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    def path2triplet(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                     if to_original_name else triplet
                str_path += (f'{list((head, relation, tail))}\n')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    def path2organize(self, paths, to_original_name=False):
        str_paths = []
        for path in paths:
            str_path = ''
            for idx, triplet in enumerate(path):
                head, relation, tail, _ = self.triplet2original_name(triplet) \
                                                     if to_original_name else triplet
                str_path += (f'{list((head, relation, tail))}\n')
                if idx == len(path) - 1:
                    prefix_score = path[idx - 1][3]
                    normalized_prefix_score = prefix_score / (len(path) - 1)
                    str_paths.append((str_path, prefix_score, normalized_prefix_score))
        return str_paths
    
    def item_information(self, iid):
        result = []
        meta_information = defaultdict(list)
        for eid, rid in self.data.train_kg_dict[iid]:
            if rid != 1:
                meta_information[rid].append(self.data.entity_id2org[eid])
        for rid, entity_list in meta_information.items():
            relation_name = self.data.relation_id2org[rid].replace('item_has_','').replace('_as_attribute:', '')
            entity_list_str = ' '.join(entity_list)
            result.append(f'The {relation_name} of the movie is/are {entity_list_str}.')
        return '\n'.join(result)
    
    def user_history(self, uid, max_items):
        result = []
        prefered_item_list = self.data.train_kg_dict[uid]
        if len(prefered_item_list) > max_items:
            prefered_item_list = random.sample(prefered_item_list, k=max_items)
        for idx, (prefered_iid, rid) in enumerate(prefered_item_list):
            result.append(f'{idx + 1}. ' + self.item_information(prefered_iid))
        return '\n'.join(result)
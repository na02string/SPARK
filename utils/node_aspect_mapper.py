# utils/node_aspect_mapper.py
import os, json
from utils.utils import llm_response
from prompts.prompt_node_aspect import node_aspect_prompt
from prompts.prompt_nodes_aspect import batch_prompt

class NodeAspectMapper:
    """
    (node, aspect) → True/False 캐시
    이제 다수 노드를 한 번에 판단합니다.
    """
    def __init__(self, cache_path: str = "cache/node_aspect_cache.jsonl"):
        self.cache_path = cache_path
        self.cache: dict[tuple[str, str], bool] = {}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                for line in f:
                    k, v = json.loads(line)
                    self.cache[tuple(k)] = v

    # --------- 배치 판단 ---------
    def judge_nodes(self, args,
                    node_names: list[str],
                    aspects: list[str]) -> dict[str, set[str]]:
        """
        여러 노드 × 여러 aspect 관계를 한 번에 판단.
        반환값: {node_name: {관련 aspect들}}
        """
        from collections import defaultdict
        name_map = {n.lower(): n for n in node_names}  # 소문자 → 원래 이름
        node_names_lc = list(name_map.keys()) # 소문자 변환

        # 캐시에 없는 (node, aspect)만 추출
        unknown_pairs = [
            (n, a) for n in node_names_lc for a in aspects
            if (n, a) not in self.cache
        ]
        yes_map: dict[str, set[str]] = {
            name_map[n]: {a for a in aspects if (n, a) in self.cache and self.cache[(n, a)]}
            for n in node_names_lc
        }
        # LLM 호출 (필요할 때만)
        if unknown_pairs:
            # (노드 → 캐시에 없는 aspect들)로 묶기
            node2aspects = defaultdict(list)
            for n, a in unknown_pairs:
                node2aspects[n].append(a) # {'a(node)': ['m', 'n'], 'b': ['l', 'm'], 'c': ['l', 'm', 'n']})
                # 캐시에 없는 조합

            # 프롬프트 생성: dict(node → [aspect list])
            prompt = batch_prompt.format(
                node_aspect_map=json.dumps(node2aspects, ensure_ascii=False)
            )

            # LLM 호출
            resp = llm_response(args, prompt)
            try:
                mapping = json.loads(resp)  # 기대: {node: [aspect, ...]}
            except json.JSONDecodeError:
                mapping = {}

            # 캐시 및 yes_map 업데이트
            for node_resp, aspect_list in mapping.items():
                n_lc = node_resp.lower()
                rel_set = set(aspect_list)
                for a in aspects:
                    key = (n_lc, a)
                    if key in self.cache:
                        continue
                                            
                    is_rel = a in rel_set
                    self._save(key, is_rel)
                    if is_rel:
                        orig_name = name_map.get(n_lc, n_lc)
                        yes_map.setdefault(orig_name, set()).add(a)

        return yes_map # {원래 node 이름: 관련 aspect 집합}

    
    def judge_node(self, args,
                    node_name: str,
                    aspects: list[str]) -> dict[str, set[str]]:
        """
        노드 × 여러 aspect 관계를 판단.
        반환값: {node_name: {관련 aspect들}}
        """
        key_base = node_name.lower()
        unknown = [a for a in aspects if (key_base, a) not in self.cache]
        yes_set = {a for a in aspects if (key_base, a) in self.cache and self.cache[(key_base, a)]}

        if unknown: # 캐시 안에 없는 경우
            prompt = node_aspect_prompt.format(node=node_name, aspect_list=json.dumps(unknown, ensure_ascii=False))
            # json.dumps(unknown, ensure_ascii=False) ex. ["storyline", "casting", "direction"]
            resp   = llm_response(args, prompt)
            try:
                related = set(json.loads(resp)) # 입력으로 준 노드랑 관련 있는 aspect들
            except json.JSONDecodeError:
                related = set()

            for asp in unknown:
                is_rel = asp in related
                self._save((key_base, asp), is_rel)
                if is_rel:
                    yes_set.add(asp)
        return yes_set

    def _save(self, key: tuple[str, str], value: bool):
        self.cache[key] = value
        with open(self.cache_path, "a") as f:
            json.dump([key, value], f, ensure_ascii=False)
            f.write("\n")

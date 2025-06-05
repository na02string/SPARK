import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from prompts.prompt_SPARK import aspect_SPARK_Prompt
from utils.aspect_extractor import extract_user_aspects
from utils.beamsearch import CollaborativeBeamSearchWithAspect 
from utils.node_aspect_mapper import NodeAspectMapper
from utils.utils import llm_response

from utils.evidence_builder import build_kg_evidence
from utils.opinion_extractor import extract_item_opinion
import json

# 사용자 맞춤 추천 설명 생성 파이프라인
#aspect-aware personalized reasoning + explanation을 모두 포함하는 함수
def get_SPARK_result(args, model, data, uid, iid):
    """
    args : 파라미터 설정 포함 (hops, num_beams, num_paths 등)
    model : 학습된 KGAT 모델
    data : DataLoaderKGAT 객체
    uid, iid : 사용자 및 추천 아이템
    """
    # user uid 와 아이템 iid에 대해 추천 설명을 생성하는 함수. 

    # 1단계: 사용자 리뷰 기반 선호 Aspect 추출
    aspect_score_dict = extract_user_aspects(args, data, uid)
    # 🆕  NodeAspectMapper 캐시 객체
    node_mapper = NodeAspectMapper(cache_path="cache/node_aspect_cache.jsonl")

    # 2단계: BeamSearch 객체 생성 (aspect score + mapper + λ 반영)
    BeamSearch = CollaborativeBeamSearchWithAspect(
                    data=data,
                    model=model,
                    user_aspect_dict=aspect_score_dict,
                    node_mapper=node_mapper,
                    lambda_aspect=args.lambda_aspect   # parser에 추가해둔 값
                    ,
                    args = args
                )

    # 3단계: 추천 경로 탐색
    save_path = {}
    
    # 다양한 hop 수 에 대해 추천 경로 추출. -> 3,5 로 설정해둠
    for hops in args.hops:
        # valid_paths : 의미 있는 경로
        valid_paths, paths = BeamSearch.search(uid, iid, 
                          only_attribute=True, remove_duplicate=True, 
                          num_beams=args.num_beams, num_hops=hops)
        
        if len(valid_paths) == 0:
            print(f"[SKIP] No valid path found for {hops} hop — skipping this hop.")
            continue        
        
        print(f"{hops} hop / valid_paths: {len(valid_paths)}, total_paths: {len(paths)}")
        save_path[hops] = BeamSearch.path2linearlize(valid_paths, to_original_name=True)
        # path2linearlize : 추출한 경로를 문자열로 바꿈. => LLM 입력용
# hop들을 모두 탐색한 후, 어떤 hop에서도 valid path가 없으면 설명 스킵
    if not save_path:
        print("[SKIP] No valid paths found for any hop — skipping explanation.")
        return None
    
    print(f"[DEBUG] save_path : {save_path}")
    
    # 4단계: 경로 선택 및 요약    
    selected_path = []
    for hops, paths in save_path.items():
        # hop 별로 상위 num_paths개의 경로만 선택. 디폴트로 4개로 해둠
        selected_path.extend([path[0] for path in paths[:args.num_paths]])
        # ['A2B73CL3QSYWLB -> user_likes_item -> 1920s_rediscovered_films -> attribute_is_subject_of_item: -> The_Beloved_Rogue -> item_has_subject_as_attribute: -> 1920s_historical_adventure_films',
        #  'A2B73CL3QSYWLB -> user_likes_item -> 1927_films -> attribute_is_subject_of_item: -> The_Beloved_Rogue -> item_has_subject_as_attribute: -> 1920s_historical_adventure_films']
    selected_path_str = '\n'.join(selected_path)

    # 5단계: LLM 기반 IC 요약 생성    
    # 첫번째 LLM 입력 생성(IC 요약 요청)
    item_information=BeamSearch.item_information(iid,max_relations=5) # item에 대한 기본 정보랑
    user_history = BeamSearch.user_history(uid, max_items=5, max_lines=10) # user가 이전에 본 item 요약(user_history)를 불러와서 
    # === 6 단계: KG evidence 생성 ===
    kg_evi = build_kg_evidence(
                args=args,
                paths=selected_path,   # 경로 문자열만
                aspect_dict=aspect_score_dict,
                data=data,
                node_mapper=node_mapper
            )

    # === 7 단계: item opinion 추출 ===
    opinions = []
    ic_json = []
    for asp, evid_list in kg_evi.items():
        if not evid_list:  # evidence가 없으면 건너뜀
            continue
        opinions = extract_item_opinion(args, data, iid, asp)
        ic_json.append({
            "aspect": asp,
            "aspect_score" : aspect_score_dict.get(asp,0.0),
            "kg_evidence": evid_list[:10],   # 최대 10개
            "item_opinion": opinions
        })
        ic_json.sort(key=lambda x: -x["aspect_score"]) # 점수 높은 순으로 정렬

    # === 5‑단계: LLM 설명 생성 ===
    IC2explanation_formatted = aspect_SPARK_Prompt.aspect_explanation_prompt.format(
                                    context=json.dumps(ic_json, ensure_ascii=False, indent=2),
                                    user=data.user_id2org[uid],
                                    item=data.entity_id2org[iid],
                                    record=user_history,
                                    item_information=item_information)
    print(f'[DEBUG] user_history {user_history}')
    print(f"[DEBUG] item_information {item_information}")
    print(f"[DEBUG] ic_json : {json.dumps(ic_json, ensure_ascii=False, indent=2)}")
    print(f"[DEBUG]kg_ev : {evid_list[:10]}")
    print(f"[DEBUG]item opinion : {opinions}")
    explanation = llm_response(args, IC2explanation_formatted)
    print(f"[DEBUG] Prompt : {IC2explanation_formatted}")
    # 로그 기록 추가
    import logging
    logging.info(f"[PROMPT] LLM Input Prompt for uid={uid}, iid={iid}:\n{IC2explanation_formatted}")

    print(f"[DEBUG] LLM Output: {explanation}")
    # 최종 추천 설명 반환
    return explanation

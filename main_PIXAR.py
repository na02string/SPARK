import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from prompts.prompt_PIXAR import PIXAR_Prompt
from utils.utils import llm_response

# kg 기반 경로 탐색 + 경로 요약 + 추천 설명 생성을 통해 user 맞춤형 설명 문장 1개를 생성하는 함수
def get_PIXAR_result(args, BeamSearch, uid, iid):
    # user uid 와 아이템 iid에 대해 추천 설명을 생성하는 함수. 
    '''
    args : 파라미터 설정 포함(hops, num_beams, num_paths 등)
    BeamSearch: kg 경로 탐색(CBS기반)
    '''
    save_path = {}
    
    # 다양한 hop 수 에 대해 추천 경로 추출. -> 3,5 로 설정해둠
    for hops in args.hops:
        # valid_paths : 의미 있는 경로
        valid_paths, paths = BeamSearch.search(uid, iid, 
                          only_attribute=True, remove_duplicate=True, 
                          num_beams=args.num_beams, num_hops=hops)
        print(f"{hops} hop / valid_paths: {len(valid_paths)}, total_paths: {len(paths)}")
        save_path[hops] = BeamSearch.path2linearlize(valid_paths, to_original_name=True)
        # path2linearlize : 추출한 경로를 문자열로 바꿈. => LLM 입력용
    
    selected_path = []
    for hops, paths in save_path.items():
        # hop 별로 상위 num_paths개의 경로만 선택. 디폴트로 4개로 해둠
        selected_path.extend([path[0] for path in paths[:args.num_paths]])
    selected_path_str = '\n'.join(selected_path)
    
    # 첫번째 LLM 입력 생성(IC 요약 요청)
    item_information=BeamSearch.item_information(iid) # item에 대한 기본 정보랑
    user_history = BeamSearch.user_history(uid, max_items=9) # user가 이전에 본 item 요약(user_history)를 불러와서 
    # path2IC_prompt에 넣어서 LLM 에게 경로 정보를 "요약된 정보(IC)" 형태로 생성하도록 요청
    path2IC_formatted = PIXAR_Prompt.path2IC_prompt.format(paths=selected_path_str,   
                                                    user=BeamSearch.data.user_id2org[uid],
                                                    item=BeamSearch.data.entity_id2org[iid],
                                                    item_information=item_information)
    
    compressed_information = llm_response(args=args, query=path2IC_formatted)
    
    # 두번째 LLM 입력 생성(추천 설명 생성)
    # 위에서 만든 IC 기반 요약이랑 사용자 리뷰 이력 등을 기반으로 실제 자연어 추천 설명을 LLM에게 요청
    IC2explanation_formatted = PIXAR_Prompt.IC2explanation_prompt.format(context=compressed_information,
                                                    user=BeamSearch.data.user_id2org[uid],
                                                    item=BeamSearch.data.entity_id2org[iid],
                                                    record=user_history,
                                                    item_information=item_information)
    explanation = llm_response(args=args, query=IC2explanation_formatted)
    
    # 최종 추천 설명 반환
    return explanation
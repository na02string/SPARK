import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT
from utils.utils import *
from utils.beamsearch import *
from utils.get_input import *

from main_SPARK import *
from main_PIXAR import *
from main_LLMXRec import *
from prompts.prompt_judge import *

import logging
from datetime import datetime

data_dir = {
    'Amazon_data_kg_2018/for_kgat_final' : '/home/nhkim02/to_do/SPARK/trained_model/KGAT/Amazon_data_kg_2018/for_kgat_final/del_low_rating_embed-dim64_relation-dim16_symmetric_gcn_64-32-16_lr0.0001_pretrain0/data_epoch30.pkl'
}
model_dir = {
    'Amazon_data_kg_2018/for_kgat_final' : '/home/nhkim02/to_do/SPARK/trained_model/KGAT/Amazon_data_kg_2018/for_kgat_final/del_low_rating_embed-dim64_relation-dim16_symmetric_gcn_64-32-16_lr0.0001_pretrain0/model_epoch30.pth'
}


def eval_llm_system(args):
    # # 로그 폴더 없으면 생성
    # os.makedirs("explain_eval_logs", exist_ok=True)
    # # 로그 파일 설정
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_filename = f"eval_log_{timestamp}.log"
    # logging.basicConfig(
    #     filename=os.path.join("explain_eval_logs", log_filename),
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s"
    # )


    
    # GPU / CPU
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    print("[INFO] Loading data...")
    # load data
    with open(data_dir[args.data_name], 'rb') as f:
        data = pickle.load(f)
    # data.df_reviews : DataFrame [user_id, text]라서 item_id가 없어서
    # data.df_reviews에 item_id 추가
    add_df = pd.read_json("/home/nhkim02/to_do/SPARK/datasets/Amazon_data_2018/Movies_and_TV_5.json", lines=True)
    # 필요한 정보만 추출
    
    print("[INFO] Data loaded")
    
    add_df_review = add_df[["reviewerID", "asin", "reviewText"]].rename(columns={
        "reviewerID": "user_id",
        "asin": "item_id",
        "reviewText": "text"
    })
    data.df_reviews = add_df_review
    print("[INFO] DataFrame reviews updated")
    
    print("[INFO] Loading model...")
    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, model_dir[args.data_name])
    model.to(device)
    print("[INFO] Model loaded")
    
    print("[INFO] Loading aspect extractor...")
    BeamSearch = CollaborativeBeamSearchWithAspect(data, model,args = args) # SPARK용 beamsearch 클래스
    BeamSearchPixar = CollaborativeBeamSearch(data, model) # pixar용 beamsearch 클래스
    
    UserItem = getUserItem(data)
    eval_results = []
    evaluation_result_path = f'./evaluation_result_compare3/hops_{args.hops}_num_llm_eval_{args.num_llm_eval}_num_paths_{args.num_paths}_judge_llm{args.judge_model}.jsonl'

    # [1] 유저 고정
    UserItem.set_uid()
    uid = UserItem.get_uid()

    # [2] 리뷰가 있는 아이템 중 이 유저가 상호작용한 것만 뽑기
    item_with_review = UserItem.get_item_with_review()
    candidate_iids = [
        iid for iid, _ in data.train_kg_dict[uid]
        if iid in item_with_review
    ]
    print(f"[INFO] user {uid}의 후보 아이템 수: {len(candidate_iids)}")

    if not candidate_iids:
        print("[ERROR] 해당 유저가 리뷰가 있는 아이템과 상호작용한 기록이 없습니다.")
        return

    # [3] 중복 없이 최대 num_llm_eval개 아이템 추출
    # unique_iids = random.sample(candidate_iids, min(args.num_llm_eval, len(candidate_iids)))
    unique_iids = candidate_iids[:args.num_llm_eval]

    # start_idx = 97
    # unique_iids = unique_iids[start_idx:]
    
    for n, iid in enumerate(unique_iids):
        print(f"[INFO] Evaluation {n+1}/{len(unique_iids)}- uid={uid}, iid={iid}")
        UserItem.uid = uid
        UserItem.iid = iid

        SPARK_explanation = get_SPARK_result(args, model, data, uid, iid)
        if SPARK_explanation is None:
            print(f"[SKIP] No explanation generated for uid={uid}, iid={iid}")
            continue
        
        results = {
            "PIXAR": get_PIXAR_result(args, BeamSearchPixar, uid, iid),
            "SPARK": get_SPARK_result(args, model, data, uid, iid),
            "LLMXRec": get_LLMXRec_result(args, BeamSearch, uid, iid)
        }
        print(f"[DEBUG] results : {results}") 
        systems = list(results.items()) # [("PIXAR", result), ("SPARK", result), ("LLMXRec", result)]
        random.shuffle(systems)
        letter_map = {"A": systems[0][0], "B": systems[1][0], "C": systems[2][0]} # {"A": "PIXAR", "B": "SPARK", "C": "LLMXRec"}
        content_map = {"A": systems[0][1], "B": systems[1][1], "C": systems[2][1]} # {"A": result, "B": result, "C": result}

        prompt = Judge_Prompt_Triple.judge_prompt.format(
            system_A=content_map["A"],
            system_B=content_map["B"],
            system_C=content_map["C"]
        )
        
        judge_response = llm_response(args=args, query=prompt, is_judge=True)
        print(f'[DEBUG] judge_response : {judge_response}')
        judge_json = extract_json_judge3(judge_response)
        print(f'[DEBUG] judge_json : {judge_json}')
        result_entry = {
            "uid": uid,
            "iid": iid,
            "letter_map": letter_map, 
            "judge_rankings": judge_json["Rankings"],
            "judge_reasons": judge_json["Reasons"],
            "overall_winner": letter_map[judge_json["Overall Winner"]],
        }
        for model_name, output in results.items():
            result_entry[f"{model_name}_result"] = output

        save_to_jsonl([result_entry], evaluation_result_path, append=True)

    print(f"[INFO] Evaluation finished. Results saved to {evaluation_result_path}")


if __name__ == '__main__':
    args = parse_kgat_args()
    setSeeds(args.seed)
    eval_llm_system(args)
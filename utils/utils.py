import random
import os
import numpy as np
import torch
from sklearn.utils import check_random_state
import openai
from openai import OpenAI
import time
import re
import json

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

# 실험 재현성을 위한 random seed 고정
def setSeeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random_state = check_random_state(seed)
    np.random.default_rng(seed=seed)

# OpenAI API 호출을 위한 메시지 포맷팅
def openai_api_messages(user_prompt):
    return [{"role": "user", "content": user_prompt}]


# OpenAI GPT 모델에 질의 → 응답 받기 + 재시도 로직
def openai_output(client, model, query):
    # 주어진 query(프롬프트)를 openai gpt 모델에 전달
    openai_input = openai_api_messages(query)
    model = model
    output = API_ERROR_OUTPUT
    # 최대 API_MAX_RETRY번까지 재시도
    for _ in range(API_MAX_RETRY):
        response = client.chat.completions.create(
            model=model,
            messages=openai_input,
            n=1,
            temperature=0,
        )
        output = response.choices[0].message.content
        try:
            response = client.chat.completions.create(
                model=model,
                messages=openai_input,
                n=1,
                temperature=0,
            )
            output = response.choices[0].message.content
            break
        except:
            print("ERROR DURING OPENAI API")
            time.sleep(API_RETRY_SLEEP)  # 에러 나면 일정 시간 대기하고 재시도
    return output # 최종적으로 .message.content값 반환.

# 	모델 종류 분기 및 API 호출 (judge_model vs llm_model)
def llm_response(args, query, is_judge=False):
    # 실제 외부에서 호출되는 메인 LLM 함수
    # args 객체에서 API키랑 사용할 모델 이름을 읽어옴
    # is_judge = True 이면 평가용 모델(ex. 평가 프롬프트), 아니면 생성 모델
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    client = OpenAI(
        api_key=os.environ.get(args.OPENAI_API_KEY),
    )
    if is_judge:
        response = openai_output(client, model=args.judge_model, query=query)
    else:
        response = openai_output(client, model=args.llm_model, query=query)
    return response

# LLM 응답 내 Json을 문자열에서 파싱
def extract_json(text):
    # 응답 텍스트 내에 {...} 형태의 JSON이 있을 경우 추출. 파싱 실패 시 약간의 수정 시도. 
    match = re.search(r'\{.*?\}', text, re.S)

    if match:
        json_content = match.group(0)
        try:
            data = json.loads(json_content.replace("'", '"'))
            return data
        except json.JSONDecodeError as e:
            fixed_json = re.sub(r',\s*\}', '}', re.sub(r',\s*\]', ']', json_content))
            try:
                data = json.loads(fixed_json)
                return data
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON again: {e}")
                return None
    else:
        print("No JSON found in the response text.")
        return None
    
def extract_json_judge3(text):
    import re, json
    try:
        # JSON 블럭만 추출 (줄바꿈 포함 허용, 가장 긴 블럭 매칭)
        match = re.search(r'\{[\s\S]*\}', text.strip())  
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode failed: {e}")
    return None


# 리스트 형태의 데이터를 .jsonl 파일로 저장
def save_to_jsonl(data, file_path, append=False):
    mode = 'a' if append else 'w'
    print(f"[DEBUG] Trying to save {len(data)} entries to {file_path}") 
    with open(file_path, mode, encoding='utf-8') as f:
        for entry in data:
            print(f"[DEBUG] Writing entry: {entry}") 
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

'''
1. extract_user_aspects(uid) 함수 작성
2. 내부에서 llm_response() 호출 → prompts/prompt_aspect.py 사용
3. 빈도수 세어 Dict[aspect:score] 리턴'''
# utils/aspect_extractor.py
import collections, json, re
from utils.utils import llm_response
from prompts.prompt_aspect import aspect_prompt
import random

def _clean(text: str) -> str:
    # 간단 전처리 (필요시 보완)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_aspects_from_reviews(args, reviews: list[str]) -> list[str]:
    """LLM 한 번 호출해 리뷰 묶음에서 aspect 배열 추출"""
    joined = "\n".join(_clean(r) for r in reviews)
    prompt = aspect_prompt.format(reviews=joined)
    resp   = llm_response(args=args, query=prompt)
    try:
        aspects = json.loads(resp)
        if isinstance(aspects, list):
            # return [a.strip() for a in aspects]
            return [a.strip() for a in aspects if isinstance(a, str) and a.strip()]
    except json.JSONDecodeError:
        pass
    # fallback: 콤마 기준 분리
    # return [w.strip() for w in resp.strip("[]").split(",") if w.strip()]
    return [w.strip() for w in resp.strip("[]").split(",") if isinstance(w, str) and w.strip()]

def extract_user_aspects(args, data, uid, top_n: int = 3, max_sentences : int = 10):
    """
    data : data.df_reviews가 있음
        df_reviews : DataFrame [user_id, item_id, text]
            user_id |item_id|text
            A2M1CU2IRZG0K9	|0005089549|So sorry I didn't purchase this years ago when...
    uid : int(user_id) ex. 52855
    Return -> Dict[str, float]  (aspect : freq/total)
    """
    df_reviews = data.df_reviews
    uid = data.user_id2org[uid] 
    user_revs = df_reviews[df_reviews["user_id"] == uid]["text"].dropna().tolist()
    # 문장 단위로 분리
    all_sentences = []
    for review in user_revs:
        sentences = re.split(r'(?<=[.!?]) +', _clean(review))
        all_sentences.extend(sentences)
        
    if len(all_sentences) == 0:
        return {}
    
    # 문장 수 제한 (랜덤 샘플링)
    sampled_sentences = random.sample(all_sentences, k=min(max_sentences, len(all_sentences)))
   
    # 1) 여러 줄 묶어서 LLM 호출 (길면 batch split)
    aspects = extract_aspects_from_reviews(args, sampled_sentences)
    if args.verbose:
        print("[Aspect LLM Output]", aspects)

    # 2) 빈도수 → 점수(비율)
    counter = collections.Counter(aspects)
    top_k = counter.most_common(top_n)
    total = sum(cnt for _, cnt in top_k) or 1
    aspect_score = {asp: round(cnt / total, 4) for asp, cnt in top_k}
    if args.verbose:
        print("[Aspect Scores]", aspect_score)
    
    if not aspect_score:
        return None


    return aspect_score
    
# 사용법 예시
# scores = extract_user_aspects(args, dataloader.df_reviews, uid=123)
# => {'연출':0.32,'감정선':0.28,'OST':0.11, ...}
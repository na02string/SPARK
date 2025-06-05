# utils/opinion_extractor.py
import random, json, re
from utils.utils import llm_response
from prompts.prompt_opinion import opinion_prompt

def sentence_split(text):
    return re.split(r'(?<=[.!?])\s+', text)

def extract_item_opinion(args, data, iid, aspect, max_sent=10):
    # 해당 아이템 리뷰 모음
    df_reviews = data.df_reviews
    iid = data.item_id2org[iid]
    revs = df_reviews[df_reviews["item_id"] == iid]["text"].dropna().tolist()
    cand  = []
    for r in revs:
        cand.extend(sentence_split(r))
    cand = [s for s in cand if 5 < len(s) < 200]
    cand = random.sample(cand, k=min(max_sent, len(cand)))
    prompt = opinion_prompt.format(item=iid, aspect=aspect, sentences="\n".join(cand))
    resp   = llm_response(args, prompt)
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return []

# utils/evidence_builder.py
# kg 경로를 자연어 evidence로 변환
from utils.node_aspect_mapper import NodeAspectMapper

def triple2sentence(head, relation, tail):
    """간단 템플릿: head (relation) tail"""
    rel_kr = relation.replace("item_has_", "").replace("_as_attribute:", "")
    return f"{head} 의 {rel_kr} → {tail}"

def build_kg_evidence(args, paths, aspect_dict, data, node_mapper):
    """
    paths         : List[str]  ("A -> r1 -> B -> r2 -> C")
    aspect_dict   : {"감정선":0.3, ...}
    Return -> Dict[aspect] = [sentence, ...]
    """
    asp_list = list(aspect_dict.keys()) # ["감정선", "연출", "OST", ...]
    grouped  = {a: [] for a in asp_list} # {"감정선": [], "연출": [], ...}

    for p in paths:
        # 삼중으로 다시 쪼개기
        seg = [s.strip() for s in p.split("->")]
        triples = [(seg[i], seg[i+1], seg[i+2]) for i in range(0, len(seg)-2, 2)]
        # p : A2B73CL3QSYWLB -> user_likes_item -> 1920s_rediscovered_films -> attribute_is_subject_of_item: -> The_Beloved_Rogue -> item_has_subject_as_attribute: -> 1920s_historical_adventure_films
        # seg : ['A2B73CL3QSYWLB', 'user_likes_item', '1920s_rediscovered_films', 'attribute_is_subject_of_item:', 'The_Beloved_Rogue', 'item_has_subject_as_attribute:', '1920s_historical_adventure_films']
        # triples : [('A2B73CL3QSYWLB', 'user_likes_item', '1920s_rediscovered_films'), ('1920s_rediscovered_films', 'attribute_is_subject_of_item:', 'The_Beloved_Rogue'), ('The_Beloved_Rogue', 'item_has_subject_as_attribute:', '1920s_historical_adventure_films')]
        # p : A2B73CL3QSYWLB -> user_likes_item -> 1927_films -> attribute_is_subject_of_item: -> The_Beloved_Rogue -> item_has_subject_as_attribute: -> 1920s_historical_adventure_films
        # seg : ['A2B73CL3QSYWLB', 'user_likes_item', '1927_films', 'attribute_is_subject_of_item:', 'The_Beloved_Rogue', 'item_has_subject_as_attribute:', '1920s_historical_adventure_films']
        # triples : [('A2B73CL3QSYWLB', 'user_likes_item', '1927_films'), ('1927_films', 'attribute_is_subject_of_item:', 'The_Beloved_Rogue'), ('The_Beloved_Rogue', 'item_has_subject_as_attribute:', '1920s_historical_adventure_films')]
        # 어떤 aspect와 관련?  (node 기준)
        path_aspects = set()
        for h, r, t in triples:
            path_aspects |= node_mapper.judge_node(args, h, asp_list) 
            path_aspects |= node_mapper.judge_node(args, t, asp_list)
        for asp in path_aspects:
            for h, r, t in triples:
                grouped[asp].append(triple2sentence(h, r, t))
    return grouped

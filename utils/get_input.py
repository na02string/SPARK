import random

class getUserItem(object):
    def __init__(self, data):
        self.uid = None
        self.iid = None
        self.data = data
        self.user_id_list = list(self.data.user_id2org.keys()) # [52855, 52856, 52857, 52858, 52859, 52860,
        self.sorted_preferences = self.get_sorted_preferences()
    
    def get_sorted_preferences(self):
        user_preferences = self.data.kg_train_data[self.data.kg_train_data.r == 0].groupby('h')['t'].count()
        sorted_preferences = user_preferences.sort_values(ascending=False)
        return sorted_preferences.to_frame().reset_index().rename(columns={"h":"uid", "t":"num_items"})
            # uid | num_items
            # 53025	| 467
            # 이런 데이터프레임 나옴
            
    def get_uid_with_review(self):
        """리뷰가 있는 uid 리스트 반환"""
        valid_reviews = self.data.df_reviews.dropna(subset=["text"])
        valid_reviews = valid_reviews[valid_reviews["text"].str.strip() != ""]
        reviewed_users = set(valid_reviews["user_id"].unique())
        return [
            uid for uid in self.user_id_list
            if self.data.user_id2org[uid] in reviewed_users
        ]
        # data.user_id2org : {52855: 'AWF2S3UNW9UA0',52856: 'A1GHUN5HXMHZ89'
        # ex. [52855, 52856, 52857, 52858, 52859, 52860, ...] # 리뷰가 있는 유저 리스트 뽑음
        
    def get_item_with_review(self):
        """리뷰가 있는 iid 리스트 반환"""
        valid_reviews = self.data.df_reviews.dropna(subset=["text"])
        valid_reviews = valid_reviews[valid_reviews["text"].str.strip() != ""] 
        reviewed_items = set(valid_reviews["item_id"].unique()) # B01HJ1INB0...
        return [
            iid for iid, org_id in self.data.item_id2org.items() # {0: '0792839072', 1: 'B0067EKYL8', 2: 'B009934S5M',
            if org_id in reviewed_items
            ] # [3, 4,...]
        
    
    def set_uid(self, is_cold_start=False, uid=None): # [52855, 52856, 52857, 5
        if is_cold_start:
            if uid == None:
                raise Exception("uid is required!")
            self.uid = uid
        else:
            uids_with_review = self.get_uid_with_review() # [52855, 52856, 52857, 52858, 52859, 52860, 
            self.uid = random.choice(uids_with_review) # 원래 이건데

    
    def set_iid(self):
        if self.uid == None:
            raise Exception("Run set_uid()")
        # user_preference_list = self.data.train_kg_dict[self.uid]
        # self.iid = random.choice(user_preference_list)[0] # 7605 같은 숫자
        item_with_review = self.get_item_with_review() # 리뷰가 있는 아이템 리스트 [2,4,3,5...]
        print(f"[DEBUG] 리뷰 있는 아이템 수: {len(item_with_review)}")
        
        candidate_iids = [
            iid for iid, _ in self.data.train_kg_dict[self.uid]  # [(0, 0), (3, 0), (4, 0), (5, 0), (6, 0), (8, 0), (9, 0), (10, 0)]
            if iid in item_with_review
        ] # 사용자 self.uid가 과거에 인터랙션한 아이템들 중에서, 리뷰가 존재하는 아이템만 필터링해서 추천 대상으로 뽑
        print(f"[DEBUG] uid={self.uid}의 후보 iid 수: {len(candidate_iids)}")
        if not candidate_iids:
            raise Exception("No valid item with review for this user.")
        self.iid = random.choice(candidate_iids)        
    
    
    def get_uid(self):
        if self.uid == None:
            raise Exception("Run set_uid()")
        return self.uid # 52855 같은 숫자
    
    def get_iid(self):
        if self.iid == None:
            raise Exception("Run set_iid()")
        return self.iid # 7605 같은 숫자
    

        
    
    def get_cold_start_uid(self, k: int = 3):
        # 인터렉션 수가 정확히 number_of_interactions인 사용자 uid를 가져옴
        cold_start_uid_list = list(self.sorted_preferences[self.sorted_preferences.num_items==k].uid)
        return cold_start_uid_list
    # [58502, 52857, 52858, 52862, 52867, 52872, ...
    
    # 새로 추가함 
    def get_review_cold_start_uid(self):
        """리뷰가 0개인 유저 ID 리스트""" # 리뷰도 콜드 스타트인거 평가해야할것 같아서
        # df_reviews는 user_id(org형, 예: 'A2M1CU2IRZG0K9')
        # 유효한 텍스트 리뷰만 필터링
        valid_reviews = self.data.df_reviews.dropna(subset=["text"]) # 리뷰가 있는 데이터만 남기기
        valid_reviews = valid_reviews[valid_reviews["text"].str.strip() != ""] # nan외에도 그냥 빈 text있는 데이터도 필터
        # 유효한 리뷰가 있는 유저의 원래 ID set
        reviewed_users_org = set(valid_reviews["user_id"].unique()) # 리뷰가 있는 유저들 집합
        # {'A34CBD9GC56BYO', 'A10JNGBPFY2DZM', ....

        # 내부 ID → org ID 변환 후 체크
        return [
            uid for uid in self.user_id_list # user_id_list는 [52855, 52856, 52857, 52858, 52859, 52860, ...]
            if self.data.user_id2org[uid] not in reviewed_users_org
        ] # ex. [52855, 52856, 52857, 52858, 52859, 52860, ...] # 리뷰가 없는 유저 리스트 뽑음

    def get_cold_start_item(self, k: int = 3):
        """인터랙션 수 <= k 인 아이템 ID 리스트"""
        df = self.data.kg_train_data
        item_cnt = df[df.r == 0].groupby("t").size() 
        return item_cnt[item_cnt <= k].index.tolist() # [3, 21,32,...

    def get_item_no_review(self):
        """리뷰가 0개인 아이템 ID 리스트"""
        valid_reviews = self.data.df_reviews.dropna(subset=["text"])
        valid_reviews = valid_reviews[valid_reviews["text"].str.strip() != ""]

        reviewed_items_org = set(valid_reviews["item_id"].unique()) # 리뷰가 있는 아이템들 집합
        # {'7502596496','B00MVIYKBY', 'B000A0GOMS', 'B000ASDFJU',
        
        return [
            iid for iid in range(self.data.n_items) # for iid in 4779
            if self.data.item_id2org[iid] not in reviewed_items_org
        ] # 리뷰가 없는 아이템 리스트 뽑음 ex. [12,566,...]


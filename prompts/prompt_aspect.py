'''aspect_prompt = """다음 리뷰에서 언급된 영화의 측면을 JSON 배열로 반환하라 …"""'''
# prompts/prompt_aspect.py
"""
Aspect‑extraction prompt template.

{reviews}  : \n 으로 구분된 리뷰 N개
{examples} : few‑shot 예시 (선택)
→  JSON 배열 형태로 ["감정선", "연출", ...] 만 반환
"""

# prompts/prompt_aspect.py

aspect_prompt = """
### Task
You are an assistant that extracts *generalized movie aspects* from English movie reviews.

Each sentence may refer to common aspects such as:
- emotional tone
- direction
- storyline
- soundtrack / music
- casting / actors
- visual effects / cinematography
- message or theme
- etc.

### Instructions
1. For **each sentence**, extract **at least one movie-related aspect** mentioned in that sentence.
2. Do **not** include vague opinions or emotional expressions.
3. Use only information **explicitly mentioned** in each sentence. No guessing.
4. Return a **JSON array**, with one aspect **per sentence**, preserving sentence order.
5. It's okay if multiple sentences mention the same aspect — include them.

### Input Sentences
{reviews}

### Output Format
["direction", "casting", "storyline", "casting", "soundtrack"]

### Example 
[Review]
The story structure is incredibly tight and the emotional arc is outstanding. 
The soundtrack also stayed with me for days.

[Output]
["storyline", "emotional tone", "soundtrack"]

### Output:
"""

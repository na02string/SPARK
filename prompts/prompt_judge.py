class Judge_Prompt(object):
    judge_prompt_comparing_2 = '''### System A:
{system_A}

### System B:
{system_B}

### Evaluation Factors:
1. User-friendliness: Evaluate which system's response is easier to understand and more relatable.
2. Clarity: Determine which system's response is more straightforward and easy to understand, using simple language, concise explanations.
3. Usefulness of Information: Assess which system's response provides more practical and relevant information that aids in decision-making and is actionable.
4. Uniqueness of Explanation: Evaluate the uniqueness of each system's recommendation explanations, giving higher ratings to those that provide particularly distinctive and non-universal explanations.
5. Innovativeness: Judge which system introduces more creative ideas or perspectives that enhance the user experience.

### Instruction:
You are an impartial judge with a professional level of knowledge about movies. System A and System B have each provided explanations for why user YM_289 received a movie recommendation for Nightmare on Elm Street 4, A - The Dream Master.
For each system, your goal is to determine which one performs better for each of the five evaluation factors listed above. Conclude which system is superior overall.
Provide your responses in the following format, where 'A' indicates System A is better, 'B' indicates System B is better, 'T' indicates that both systems A and B are similar, resulting in a draw. (Example 1: ['T', 'B', 'A', 'T', 'B'], Example 2: ['B', 'A', 'B', 'A', 'B'], Example 3: ['B', 'T', 'B', 'B', A], Example 4: ['A', 'B', 'T', 'B', 'T'])

Your response must be formatted as JSON:
{{
    "Reasons": "Brief reasons for the judgment",
    "Answer": [results]
}}

Let's think step by step!

### Response:
'''

class Judge_Prompt_Triple(object):
    judge_prompt = """
### System A:
{system_A}

### System B:
{system_B}

### System C:
{system_C}

### Evaluation Factors:
1. User-friendliness: Evaluate which system's response is easier to understand and more relatable.
2. Clarity: Determine which system's response is more straightforward and easy to understand, using simple language, concise explanations.
3. Usefulness of Information: Assess which system's response provides more practical and relevant information that aids in decision-making and is actionable.
4. Uniqueness of Explanation: Evaluate the uniqueness of each system's recommendation explanations, giving higher ratings to those that provide particularly distinctive and non-universal explanations.
5. Innovativeness: Judge which system introduces more creative ideas or perspectives that enhance the user experience.


### Instructions:
You are an impartial judge with professional knowledge of movie recommendation systems.  
You are given **three explanations** generated by different systems for why a user should be recommended a movie.  
Evaluate the quality of these explanations for each of the five evaluation factors listed above.

For **each evaluation factor**, rank the systems from **best to worst (1st to 3rd)**.  
**Do not allow ties.**

### Format your response strictly as JSON:
{{
  "Rankings": {{
    "User-friendliness": ["A", "B", "C"],
    ...
  }},
  "Reasons": {{
    "User-friendliness": "...",
    ...
  }},
  "Overall Winner": "A",
  "Overall Reason": "..."
}}

Let’s begin step-by-step.

### Response:
"""

# '''{
#   "Rankings": {
#     "User-friendliness": ["B", "A", "C"],
#     "Clarity": ["A", "B", "C"],
#     "Usefulness of Information": ["B", "C", "A"],
#     "Uniqueness of Explanation": ["C", "B", "A"],
#     "Innovativeness": ["C", "A", "B"]
#   },
#   "Reasons": {
#     "User-friendliness": "System B uses a friendly and relatable tone, making it easier to connect with. System A is also readable but more formal, while System C feels abstract.",
#     "Clarity": "System A offers the clearest structure and flow of explanation. System B is also clear but slightly verbose. System C is less straightforward.",
#     "Usefulness of Information": "System B provides specific, actionable insights about the film. System C gives some context, while System A is more general.",
#     "Uniqueness of Explanation": "System C introduces novel elements in its reasoning. System B offers some personalization, but System A is more generic.",
#     "Innovativeness": "System C proposes a unique narrative framing. System A and B use more conventional explanation styles."
#   },
#   "Overall Winner": "B",
#   "Overall Reason": "System B consistently ranks high in clarity, usefulness, and relatability, making it the most effective for users."
# } -> 예상 응답
# '''
# prompts/prompt_opinion.py

opinion_prompt = """
### Task
You are an assistant that selects opinion sentences related to a specific movie aspect from a list of review sentences for the movie <{item}>.

### Aspect
"{aspect}"

### Instructions
1. From the list of review sentences below, select **only the ones that express opinions directly related to the above aspect**.
2. A valid opinion sentence should include a subjective evaluation or sentiment (positive or negative) about the aspect.
3. Do **not** select sentences that are off-topic, vague, or purely factual without an opinion.
4. Return **up to 6 sentences**, exactly as written, in a **JSON array**.
5. Do **not** add any explanations, summaries, or formatting outside the JSON.

### Input Sentences
{sentences}

### Output Format
Example:
[
  "The soundtrack was haunting and unforgettable.",
  "I loved how the music added depth to the emotional scenes.",
  ...
]

### Output:
"""

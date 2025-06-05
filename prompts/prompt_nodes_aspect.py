# prompts/prompt_nodes_aspect.py

batch_prompt = """
### Task
You are an assistant that determines **which aspects from a given list are directly relevant to each knowledge-graph node**.

### Mapping
You are given a JSON object. Each key is a node name, and the corresponding value is a list of aspect candidates.

{node_aspect_map}

### Instructions
1. For **each node**, select the aspects that are clearly and semantically related to that node in the movie domain.
2. Use common sense, world knowledge, and cultural context when necessary.
3. **Do not include aspects that are weakly, indirectly, or ambiguously related.**
4. Respond only with a single valid JSON object. **No explanations or extra text.**

### Output Format
Example:
{{
  "Christopher Nolan": ["direction", "visuals"],
  "Hans Zimmer":       ["soundtrack"]
}}

### Output:
"""

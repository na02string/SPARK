# prompts/prompt_node_aspect.py

node_aspect_prompt = """
You are an assistant that identifies which aspects are relevant to a given movie-related node in a knowledge graph.

### Task
You are given a node from a movie knowledge graph and a list of candidate aspects (e.g., storyline, casting, direction, music). Your job is to determine which of these aspects are semantically relevant to the node.

- Return a JSON array containing only the aspects that are clearly related to the node.
- Do not guess. If no aspect clearly applies, return an empty list.
- Only select aspects that would commonly be associated with the type or meaning of the node.
- Do not add explanations. Return only a JSON array.

### Node
{node}

### Candidate Aspects
{aspect_list}

### Response:
"""

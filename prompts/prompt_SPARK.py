class aspect_SPARK_Prompt(object):    
    aspect_explanation_prompt = """
### Task:
You are an intelligent and friendly movie recommendation assistant. Your job is to explain **why a specific user will likely enjoy a particular movie**, based on their **individual preferences, past viewing history, and aspect reasoning**.

You are given:
- A **target movie**, with metadata (title, genre, actors, year, directors)
- The **user’s past liked movies** and their metadata
- **Aspect-based reasoning evidence**, including:
  - User's **aspect preference scores** (e.g., storyline, acting)
  - Knowledge graph-based evidence (kg_evidence)
  - Opinionated review sentences (item_opinion)

### Instructions:
Write a single **paragraph** with **exactly 3 clear and friendly reasons** why the user might enjoy the movie.

Each reason must:
- Be based on one of the user's top-rated aspects.
- Mention either:
  - A movie the user liked in the past, OR
  - A fact from kg_evidence or item_opinion.
- Be **easy to understand, specific, and natural**—like you're talking to a person.
- Focus on the **similarities or connections** between this movie and their past favorites.

### Writing Style:
- Use natural, friendly language.
- Keep sentences short and clear.
- Avoid jargon.
- No bullet points or numbering.
- Keep it under **150 words**.

### Example:
You’re likely to enjoy *Casablanca* because it shares a deep emotional storyline similar to *The Notebook*, which you loved. The way characters express love and sacrifice will resonate with you. Plus, its classic black-and-white visuals match the nostalgic tone you enjoyed in *Roman Holiday*.

### Movie:
Title: {item}
{item_information}

### User Viewing History:
{record}

### Aspect Reasoning Context:
{context}

### Now write your response:
"""
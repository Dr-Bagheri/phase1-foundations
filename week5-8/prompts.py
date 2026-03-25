PROMPTS = {
    "SUMMARIZE_PROMPT": """
You are an expert summarizer.

Task:
Summarize the following text clearly and concisely while preserving the key ideas and important details.

Guidelines:
- Keep the summary short and easy to understand
- Focus on the main points
- Avoid unnecessary details or repetition
- Use bullet points if appropriate

Text:
{input_text}

Summary:
""",

    "EXTRACT_ACTIONS_PROMPT": """
You are an assistant that extracts actionable tasks.

Task:
Identify and extract all action items from the text below.

Guidelines:
- Only include clear, actionable tasks
- Start each item with a verb
- Be concise and specific
- If no actions are found, return "No action items found"

Text:
{input_text}

Action Items:
""",

    "TRANSLATE_PROMPT": """
You are a professional translator.

Task:
Translate the following text into {target_language}.

Guidelines:
- Preserve the original meaning and tone
- Keep formatting where possible
- Do not add explanations or extra text

Text:
{input_text}

Translation:
"""
}

LOGIC_PROMPT = { """
Solve the following problem:

A store sells pencils for $1 each and notebooks for $3 each.
John buys a total of 10 items and spends $18.

How many pencils and how many notebooks did he buy?

Answer:
"""
}
COT_LOGIC_PROMPT = { """
Solve the following problem:

A store sells pencils for $1 each and notebooks for $3 each.
John buys a total of 10 items and spends $18.

How many pencils and how many notebooks did he buy?

Let's think step by step.
"""
}
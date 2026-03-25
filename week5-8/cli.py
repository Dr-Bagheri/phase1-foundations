import argparse
from llm_client import LLMClient
from prompts import PROMPTS
import os
import sys


def load_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_prompt(task_name, text, target_language=None):
    task_map = {
        "summarize": "SUMMARIZE_PROMPT",
        "extract_actions": "EXTRACT_ACTIONS_PROMPT",
        "translate": "TRANSLATE_PROMPT"
    }

    if task_name not in task_map:
        print(f"Error: Unsupported task '{task_name}'")
        print(f"Supported tasks: {list(task_map.keys())}")
        sys.exit(1)

    template = PROMPTS[task_map[task_name]]

    if task_name == "translate":
        if not target_language:
            print("Error: --target_language is required for translation")
            sys.exit(1)
        return template.format(input_text=text, target_language=target_language)

    return template.format(input_text=text)


def main():
    parser = argparse.ArgumentParser(description="LLM CLI Tool")

    parser.add_argument("--file", required=True, help="Path to input text file")
    parser.add_argument("--task", required=True,
                        choices=["summarize", "extract_actions", "translate"],
                        help="Task to perform")
    parser.add_argument("--target_language", help="Target language (for translation only)")

    args = parser.parse_args()

    text = load_file(args.file)

    prompt = get_prompt(args.task, text, args.target_language)

    client = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))

    print("\n Processing...\n")
    result = client.generate(prompt)

    print("=== RESULT ===\n")
    print(result)


if __name__ == "__main__":
    main()
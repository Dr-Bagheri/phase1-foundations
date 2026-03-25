from openai import OpenAI

class LLMClient:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, model="gpt-4.1-mini"):
        response = self.client.responses.create(
            model=model,
            input=prompt
        )
        return response.output[0].content[0].text
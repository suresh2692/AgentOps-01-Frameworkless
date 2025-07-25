import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class AzureOpenAIClient:
    def __init__(self):
        self.model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_API_ENDPOINT'),
            api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        )

    def generate_response(self, messages, tools):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages, tools= tools,
            max_tokens=150
        )
        return response
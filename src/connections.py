"""Connections to OpenAI and Azure OpenAI clients."""
import os

def get_azure_openai_client():
    
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    AZURE_ENDPOINT = "https://models.github.ai/inference"
    AZURE_TOKEN = os.getenv("GITHUB_OPENAI_API_KEY")
    if not AZURE_TOKEN:
        err_msg = "AZURE_TOKEN environment variable not set."
        raise ValueError(err_msg)
    client = ChatCompletionsClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_TOKEN),
    )
    return client

def get_openai_client():
    from openai import OpenAI
    OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_TOKEN)
    return client
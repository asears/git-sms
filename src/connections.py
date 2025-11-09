"""Connections to OpenAI and Azure OpenAI clients."""

import os
from typing import Any


def get_azure_openai_client() -> Any:
    """Get Azure OpenAI client.

    Raises:
        ValueError: If AZURE_TOKEN environment variable is not set.

    Returns:
        ChatCompletionsClient: Azure OpenAI client.
    """
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential

    azure_endpoint = "https://models.github.ai/inference"
    azure_token = os.getenv("GITHUB_OPENAI_API_KEY")

    if not azure_token:
        err_msg = "AZURE_TOKEN environment variable not set."
        raise ValueError(err_msg)
    return ChatCompletionsClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_token),
    )


def get_openai_client() -> Any:
    """Get OpenAI client.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.

    Returns:
        OpenAI: OpenAI client.
    """
    from openai import OpenAI

    openai_token = os.getenv("OPENAI_API_KEY")
    if not openai_token:
        err_msg = "OPENAI_API_KEY environment variable not set."
        raise ValueError(err_msg)
    return OpenAI(api_key=openai_token)

"""Git SMS - FastAPI app for GitHub SMS interactions."""

import html
import os
from typing import Annotated

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import Response

from commands import create_issue, create_repo
from natural_language_router import route_natural_command
from summarizers import summarize_any_repo, summarize_latest_issue, summarize_specific_issue

SLICE_LIMIT = 4000  # Max README slice size
ASK_OPENAI_MAX_LENGTH = 1000  # Max response length

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    # Initialize FastAPI app
    app = FastAPI()

    # Azure OpenAI configuration
    endpoint = "https://models.github.ai/inference"
    model_name = "openai/gpt-4o"
    token = os.getenv("GITHUB_OPENAI_API_KEY")

    if not token:
        err_msg = "GITHUB_OPENAI_API_KEY environment variable not set."
        raise ValueError(err_msg)

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )


def ask_openai(prompt: str, temperature: float = 0.7) -> str:
    """Ask the OpenAI model to generate a summary based on the prompt.

    Args:
        prompt (str): The prompt to send to the AI model.
        temperature (float): The sampling temperature for response variability.

    Returns:
        str: The generated summary or an error message.
    """
    prompt += "\n\nLimit the summary to no more than 1,000 characters. Return plain text only. No formatting."
    print("\n[ask_openai] Prompt being sent:\n", prompt)

    try:
        response = client.complete(  # type: ignore[unresolved-attribute]
            messages=[
                SystemMessage("You are a helpful assistant."),
                UserMessage(prompt),
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=700,
            model=model_name,
        )
        if not response or not response.choices:
            print("[ask_openai] No choices returned.")
            return "Sorry, no summary was generated."

        result = response.choices[0].message.content.strip()
        print(f"[ask_openai] Response length: {len(result)}")

        if len(result) > ASK_OPENAI_MAX_LENGTH:
            print("[ask_openai] Response too long.")
            return "The summary was too long to send. Try summarizing a smaller repo or issue."

        return result

    except Exception as e:
        print("[ask_openai] Exception:", e)
        return "AI summarization failed. Please try again later."


@app.post("/webhook")
async def sms_webhook(From: Annotated[str, Form()], Body: Annotated[str, Form()]) -> Response:  # noqa: N803
    """Handle incoming SMS webhook from Twilio.

    Args:
        From (str): The sender's phone number.
        Body (str): The text message body.

    Returns:
        Response: TwiML XML response.
    """
    text = Body.strip()
    print(f"\n[webhook] Incoming from {From}: {text}")
    parts = text.split()

    response_message: str | None = None

    if text.lower().startswith("create repo") and len(parts) >= 3:
        repo_name = parts[2]
        success = create_repo(repo_name)
        response_message = f"Created repo '{repo_name}'" if success else f"Failed to create repo '{repo_name}'."

    elif text.lower().startswith("create issue") and len(parts) >= 4:
        try:
            pre, body = text.split(" -- ", 1)
            _, _, repo, *title_parts = pre.split()
            title = " ".join(title_parts)
            success = create_issue(repo, title, body)
            response_message = f"Issue created in '{repo}'" if success else f"Failed to create issue in '{repo}'."
        except ValueError:
            response_message = "Usage: create issue <repo> <title> -- <body>"

    elif text.lower() == "help":
        response_message = "Available commands:\n- summarize owner/repo\n- summarize owner/repo issue [#]"

    elif text.lower().startswith("summarize") and len(parts) >= 2 and "/" in parts[1]:
        owner_repo = parts[1]
        if len(parts) >= 4 and parts[2].lower() == "issue" and parts[3].isdigit():
            issue_number = int(parts[3])
            summary = summarize_specific_issue(owner_repo, issue_number)
            response_message = summary or "Could not summarize issue."
        elif len(parts) >= 3 and parts[2].lower() == "issue":
            summary = summarize_latest_issue(owner_repo)
            response_message = summary or "Could not summarize the latest issue."
        else:
            summary = summarize_any_repo(owner_repo)
            response_message = summary or "Could not summarize that repo."

    else:
        # 🔁 Fall back to natural language routing
        print("[fallback] Attempting natural language parse")
        fallback = route_natural_command(text, model_name)
        if fallback:
            response_message = fallback

    if response_message is None:
        response_message = "Unrecognized command. Text 'help' for available options."

    return twilio_reply(response_message)


def get_headers() -> dict[str, str]:
    """Get headers for GitHub API requests, using authentication if available.

    Only read-only access with no authentication token is supported for now.

    Returns:
        dict: Headers for GitHub API requests.
    """
    return {"Accept": "application/vnd.github+json"}


def twilio_reply(message: str) -> Response:
    """
    Generate a TwiML XML response for Twilio SMS reply.

    Args:
        message (str): The message to send back.

    Returns:
        Response: FastAPI Response with TwiML XML content.
    """
    print("[twilio_reply] Responding with message:", message)
    escaped = html.escape(message or "No content.")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{escaped}</Message>
</Response>"""
    return Response(content=xml, media_type="application/xml")

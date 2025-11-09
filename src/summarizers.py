"""Summarization utilities using OpenAI or Azure OpenAI."""

import base64
import logging
import os
from typing import Literal
import tiktoken
import httpx
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from commands import get_headers
from connections import get_azure_openai_client, get_openai_client

load_dotenv()

# Setup logging
log = logging.getLogger(__name__)

# OpenAI configuration (Azure preferred if available)
USE_AZURE = bool(os.getenv("GITHUB_OPENAI_API_KEY"))
MAX_TOKENS = 16000
SLICE_LIMIT = 4000  # Max README slice size
ASK_OPENAI_MAX_LENGTH = 1000  # Max response length
AZ_OPENAI_MODEL_NAME = "openai/gpt-4o"
MODEL_NAME = "gpt-4o"

if USE_AZURE:
    client = get_azure_openai_client()
else:
    client = get_openai_client()

def get_encoder(encoding_name="cl100k_base") -> tiktoken.Encoding:
    """Get the appropriate tokenizer/encoder based on the model being used.

    Args:
        encoding_name (str): The name of the encoding to use.
        
    Returns:
        tiktoken.Encoding: The tokenizer/encoder.
    """
    encoder = tiktoken.get_encoding(encoding_name)
    return encoder


def num_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Calculate the number of tokens in the given text.

    Args:
        text (str): The input text.
        encoder (tiktoken.Encoding): The tokenizer/encoder to use.

    Returns:
        int: The number of tokens.
    """
    
    return len(encoder.encode(text))


def truncate_text(text: str, token_limit: int, encoder: tiktoken.Encoding) -> str:
    """Truncate text to fit within the specified token limit.

    Args:
        text (str): The input text.
        token_limit (int): The maximum number of tokens allowed.
        encoder (tiktoken.Encoding): The tokenizer/encoder to use.
        
    Returns:
        str: The truncated text.
    """
    tokens = encoder.encode(text)
    return encoder.decode(tokens[:token_limit])


def ask_openai(prompt: str) -> None:
    """Ask OpenAI or Azure OpenAI to summarize the given prompt.

    Args:
        prompt (str): The prompt to summarize.

    Returns:
        str | None: The summary returned by the model, or None on failure.
    """
    try:
        if USE_AZURE:
            response = client.complete(  # type: ignore[unresolved-attribute]
                model=AZ_OPENAI_MODEL_NAME,
                temperature=0.2,
                max_tokens=700,
                messages=[
                    SystemMessage(content="Summarize GitHub repo content in under 1000 characters."),
                    HumanMessage(content=prompt),
                ],
            )
            return response.choices[0].message.content.strip()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role": "system", "content": "Summarize GitHub repo content in under 1000 characters."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("[ask_openai] Error: %s", e)
        return None


def summarize_any_repo(repo_full_name: str) -> str:
    """Summarize any GitHub repository using its full name.

    Args:
        repo_full_name (str): The full name of the repository (e.g., "owner/repo").

    Returns:
        str: The summary of the repository.

    """
    encoder = get_encoder()
    print(f"[summarize_any_repo] Summarizing {repo_full_name}")
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}

    repo_url = f"https://api.github.com/repos/{repo_full_name}"
    readme_url = f"{repo_url}/readme"

    try:
        repo_res = httpx.get(repo_url, headers=headers)
        readme_res = httpx.get(readme_url, headers=headers)

        if not repo_res.is_success:
            return "Could not find repo."

        repo = repo_res.json()
        readme = readme_res.json().get("content", "") if readme_res.is_success else ""

        readme = readme.encode("utf-8")

        readme_text = base64.b64decode(readme).decode("utf-8", errors="ignore")

        prompt = f"""
Summarize this GitHub repo:

Repo Name: {repo.get("name")}
Owner: {repo.get("owner", {}).get("login")}
Description: {repo.get("description")}
Stars: {repo.get("stargazers_count")}
Forks: {repo.get("forks_count")}
Primary Language: {repo.get("language")}

README:
{readme_text}
        """.strip()

        total_tokens = num_tokens(prompt, encoder)
        if total_tokens > MAX_TOKENS:
            print(f"[summarize_any_repo] Trimming README to fit {MAX_TOKENS} token budget.")
            allowable_tokens = MAX_TOKENS - num_tokens(prompt, encoder) + num_tokens(readme_text, encoder)
            readme_text = truncate_text(readme_text, allowable_tokens)
            prompt = f"""
Summarize this GitHub repo:

Repo Name: {repo.get("name")}
Owner: {repo.get("owner", {}).get("login")}
Description: {repo.get("description")}
Stars: {repo.get("stargazers_count")}
Forks: {repo.get("forks_count")}
Primary Language: {repo.get("language")}

README:
{readme_text}
            """.strip()

        summary = ask_openai(prompt)
        return summary or "AI summarization failed."

    except Exception as e:
        log.error("[summarize_any_repo] Error: %s", e)
        return "Something went wrong."


def summarize_latest_issue(
    repo_full_name: str,
) -> Literal["AI summarization failed.", "Failed to summarize issue.", "No issues found."]:
    """Summarize the latest issue in a GitHub repository.

    Args:
        repo_full_name (str): The full name of the repository (e.g., "owner
        /repo").

    Returns:
        str: The summary of the latest issue.
    """
    print(f"[summarize_latest_issue] Summarizing latest issue in {repo_full_name}")
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}

    issues_url = f"https://api.github.com/repos/{repo_full_name}/issues"
    try:
        res = httpx.get(issues_url, headers=headers, params={"state": "open", "per_page": 1})
        if not res.is_success or not res.json():
            return "No issues found."
        issue = res.json()[0]

        issue_text = f"Issue #{issue['number']}: {issue['title']}\n{issue.get('body', '')}"
        prompt = f"""
            Summarize this GitHub issue thread:\n{issue_text}\n\nL
            imit the summary to no more than 1,000 characters.
            Return plain text only. No formatting.
        """.strip()

        summary = ask_openai(prompt)
        return summary or "AI summarization failed."
    except Exception as e:
        log.error("[summarize_latest_issue] Error: %s", e)
        return "Failed to summarize issue."


# def summarize_any_repo(owner_repo: str) -> str | None:
#     """
#     Summarize any GitHub repository given its owner/repo string.

#     Args:
#         owner_repo (str): The repository in the format 'owner/repo'.

#     Returns:
#         str | None: The summary of the repository or None if not found.
#     """
#     try:
#         owner, repo = owner_repo.split("/")
#     except ValueError:
#         return "Invalid format. Use <owner>/<repo>"

#     readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
#     repo_url = f"https://api.github.com/repos/{owner}/{repo}"
#     headers = {"Accept": "application/vnd.github+json"}

#     readme_res = httpx.get(readme_url, headers=headers)
#     repo_res = httpx.get(repo_url, headers=headers)

#     if not readme_res.is_success or not repo_res.is_success:
#         return None

#     try:
#         download_url = readme_res.json().get("download_url")
#         readme_content = httpx.get(download_url).text if download_url else "No README found."
#         if len(readme_content) > SLICE_LIMIT:
#             readme_content = readme_content[:4000]
#     except Exception as e:
#         print("[summarize_any_repo] Error downloading README:", e)
#         readme_content = "No README content available."

#     repo_data = repo_res.json()
#     prompt = f"""
# Summarize this GitHub repo:

# Repo Name: {repo_data.get("name") or "Unknown"}
# Owner: {owner}
# Description: {repo_data.get("description") or "No description"}
# Stars: {repo_data.get("stargazers_count", 0)}
# Forks: {repo_data.get("forks_count", 0)}
# Primary Language: {repo_data.get("language") or "Unknown"}

# README:
# {readme_content}
# """
#     return ask_openai(prompt)


# def summarize_latest_issue(owner_repo: str) -> str | None:
#     """
#     Summarize the latest issue in the specified GitHub repository.

#     Args:
#         owner_repo (str): The repository in the format 'owner/repo'.

#     Returns:
#         str | None: The summary of the latest issue or None if not found.
#     """
#     try:
#         owner, repo = owner_repo.split("/")
#     except ValueError:
#         return "Invalid format. Use <owner>/<repo>"

#     issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
#     res = httpx.get(issues_url, headers=get_headers(), params={"state": "open", "per_page": 1})
#     if not res.is_success or not res.json():
#         print("[summarize_latest_issue] Issue fetch failed:", res.status_code)
#         return None

#     issue = res.json()[0]
#     return summarize_issue_thread(owner, repo, issue["number"])


def summarize_specific_issue(owner_repo: str, issue_number: int) -> str | None:
    """
    Summarize a specific issue in the given GitHub repository.

    Args:
        owner_repo (str): The repository in the format 'owner/repo'.
        issue_number (int): The issue number to summarize.

    Returns:
        str | None: The summary of the specified issue or None if not found.
    """
    try:
        owner, repo = owner_repo.split("/")
    except ValueError:
        return "Invalid format. Use <owner>/<repo>"

    return summarize_issue_thread(owner, repo, issue_number)


def summarize_issue_thread(owner: str, repo: str, issue_number: int) -> str | None:
    """
    Summarize the thread of a specific issue in a GitHub repository.

    Args:
        owner (str): The repository owner.
        repo (str): The repository name.
        issue_number (int): The issue number to summarize.

    Returns:
        str | None: The summary of the issue thread or None if not found.
    """
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    issue_res = httpx.get(issue_url, headers=get_headers())
    comments_res = httpx.get(comments_url, headers=get_headers())

    if not issue_res.is_success or not comments_res.is_success:
        print("[summarize_issue_thread] Failed to fetch issue or comments.")
        return None

    issue = issue_res.json()
    thread = f"Issue #{issue_number}: {issue.get('title')}\n{issue.get('body', '')}"
    for comment in comments_res.json():
        thread += "\n" + comment.get("body", "")

    prompt = f"Summarize this GitHub issue thread:\n{thread}"
    return ask_openai(prompt)

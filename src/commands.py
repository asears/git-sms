"""GitHub API related utilities."""

import httpx


def get_headers() -> dict[str, str]:
    """
    Get headers for GitHub API requests, using authentication if available.

    Only read-only access with no authentication token is supported for now.

    Returns:
        dict: Headers for GitHub API requests.
    """
    return {"Accept": "application/vnd.github+json"}


def get_authenticated_username() -> str | None:
    """Get the username of the authenticated user, if any.

    Returns:
        str | None: The username of the authenticated user, or None if not authenticated.
    """
    user_url = "https://api.github.com/user"
    user_res = httpx.get(user_url, headers=get_headers())

    if not user_res.is_success:
        return None

    return str(user_res.json().get("login"))


def create_issue(repo: str, title: str, body: str) -> bool:  # noqa: ARG001
    """
    Create a new issue in the specified GitHub repository.

    Args:
        repo (str): The repository in the format 'owner/repo'.
        title (str): The title of the issue.
        body (str): The body content of the issue.

    Returns:
        bool: True if the issue was created successfully, False otherwise.
    """
    # Write operations are disabled in this prototype.
    print("[create_issue] Disabled: write operations are disabled in this prototype.")
    return False


def create_repo(name: str) -> bool:  # noqa: ARG001
    """Create a new GitHub repository with the given name.

    Write operations are disabled in this prototype because it is read-only.

    Args:
        name (str): The name of the repository to create.

    Returns:
        bool: True if the repository was created successfully, False otherwise.
    """

    print("[create_repo] Disabled: write operations are disabled in this prototype.")
    return False

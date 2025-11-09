"""Test summarizers."""


def test_summarize_text(mocker) -> None:
    """
    Test text summarization.

    Asserts:
        - The summary returned is as expected.
    """
    mocker.patch("connections.get_openai_client")
    mocker.patch("tiktoken.get_encoding")
    from summarizers import summarize_any_repo

    repo_full_name = "octocat/Hello-World"
    summary = summarize_any_repo(repo_full_name=repo_full_name)

    assert summary == "This is a summary."

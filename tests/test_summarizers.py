"""Test summarizers."""



def test_summarize_text(mocker) -> None:
    """
    Test text summarization.

    Asserts:
        - The summary returned is as expected.
    """
    mocker.patch("connections.get_openai_client")
    from summarizers import summarize_text
    
    text = "This is a long text that needs to be summarized."
    summary = summarize_text(text)

    assert summary == "This is a summary."

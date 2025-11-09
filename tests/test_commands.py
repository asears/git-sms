"""Test commands."""

from commands import get_authenticated_username


def test_get_authenticated_username(mocker) -> None:
    """
    Test getting the authenticated username.

    Asserts:
        - The username returned is as expected.
    """
    mocker.patch("commands.requests.get", return_value=mocker.Mock(ok=True, json=lambda: {"login": "testuser"}))

    username = get_authenticated_username()

    assert username == "testuser"

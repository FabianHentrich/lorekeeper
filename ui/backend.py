"""Backend responses on the UI pages: one error path for every page.

``requests``' own ``raise_for_status()`` says "400 Client Error: Bad Request for url …"; the
backend's ``detail`` (German, meant for the user) is lost. ``raise_for_status`` here raises a
``BackendError`` carrying that detail and the status code, so a page can show the reason or
branch on the code (e.g. 409 = a job is already running).
"""
import requests


class BackendError(RuntimeError):
    """Non-2xx answer of the backend; ``str()`` is "<code>: <detail>"."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def raise_for_status(resp: requests.Response) -> None:
    """Raise ``BackendError`` with the backend's ``detail`` for a non-2xx response.

    A Pydantic 422 carries a list of errors as ``detail`` — only their messages are shown.
    """
    if resp.ok:
        return
    try:
        body = resp.json()
        detail = body.get("detail", resp.text) if isinstance(body, dict) else resp.text
    except ValueError:
        detail = resp.text
    if isinstance(detail, list):
        detail = "; ".join(str(err.get("msg", err)) if isinstance(err, dict) else str(err) for err in detail)
    raise BackendError(resp.status_code, str(detail))

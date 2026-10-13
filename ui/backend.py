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
        """Keep code and detail separately so pages can branch on the code."""
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


def get_json(url: str, timeout: float = 5, **kwargs):
    """GET, status check, JSON — the read path of every page.

    Without the check a 4xx/5xx body (``{"detail": ...}``) was used as data and failed later as a
    ``KeyError`` or an empty list instead of showing the backend's reason (roadmap 11.0 P5).
    Raises ``BackendError`` (non-2xx) or a ``requests`` exception (backend unreachable).
    """
    resp = requests.get(url, timeout=timeout, **kwargs)
    raise_for_status(resp)
    return resp.json()

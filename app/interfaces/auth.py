from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Request


def require_token(request: Request,
                  authorization: str | None = Header(default=None)) -> None:
    token = os.getenv("API_TOKEN")
    if not token:
        return
    provided = authorization or ""
    if not provided.startswith("Bearer "):
        query_token = request.query_params.get("token")
        if query_token:
            provided = f"Bearer {query_token}"
    if not hmac.compare_digest(provided, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Unauthorized")

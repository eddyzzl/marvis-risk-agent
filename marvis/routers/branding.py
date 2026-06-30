from fastapi import APIRouter, Request

from marvis.branding import load_branding


router = APIRouter(prefix="/api", tags=["branding"])


@router.get("/branding")
def get_branding(request: Request) -> dict[str, object]:
    return load_branding(request.app.state.settings.workspace)

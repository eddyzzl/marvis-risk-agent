import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from marvis.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _edge_pixels(image: Image.Image):
    width, height = image.size
    yield from (image.getpixel((x, 0)) for x in range(width))
    yield from (image.getpixel((x, height - 1)) for x in range(width))
    yield from (image.getpixel((0, y)) for y in range(height))
    yield from (image.getpixel((width - 1, y)) for y in range(height))


def test_default_marvis_logo_assets_are_transparent_squares():
    for relative_path in [
        "marvis/static/brand/marvis-logo.png",
        "marvis/static/brand/marvis-favicon.png",
        "marvis/static/brand/marvis-workspace-logo.png",
        "marvis/static/brand/marvis-chat-logo.png",
    ]:
        image = Image.open(PROJECT_ROOT / relative_path).convert("RGBA")
        width, height = image.size
        corners = [
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        ]

        assert width == height
        assert all(pixel[3] == 0 for pixel in corners)
        assert any(pixel[3] == 0 for pixel in _edge_pixels(image))
        assert image.getchannel("A").getextrema()[1] == 255


def test_default_browser_chrome_icons_have_mode_specific_backgrounds():
    expected_sizes = {
        "marvis/static/brand/marvis-favicon-dark.png": 128,
        "marvis/static/brand/marvis-apple-touch-icon.png": 180,
        "marvis/static/brand/marvis-apple-touch-icon-dark.png": 180,
        "marvis/static/brand/marvis-app-icon-192.png": 192,
        "marvis/static/brand/marvis-app-icon-512.png": 512,
    }
    for relative_path, size in expected_sizes.items():
        image = Image.open(PROJECT_ROOT / relative_path).convert("RGBA")
        assert image.size == (size, size)
        assert image.getchannel("A").getextrema()[1] == 255

    dark_icon = Image.open(PROJECT_ROOT / "marvis/static/brand/marvis-favicon-dark.png").convert("RGBA")
    light_icon = Image.open(PROJECT_ROOT / "marvis/static/brand/marvis-apple-touch-icon.png").convert("RGBA")

    assert dark_icon.getpixel((64, 4))[:3] == (24, 24, 24)
    assert light_icon.getpixel((90, 4))[:3] == (245, 245, 247)


def test_branding_defaults_to_public_marvis_without_config(tmp_path: Path):
    client = _client(tmp_path)

    response = client.get("/api/branding")

    assert response.status_code == 200
    assert response.json() == {
        "platformName": "MARVIS-全能风控智能体",
        "browserTitle": "MARVIS-全能风控智能体",
        "primaryColor": "#303034",
        "logoUrl": "static/brand/marvis-workspace-logo.png?v=20260624-gauge",
        "workspaceLogoUrl": "static/brand/marvis-workspace-logo.png?v=20260624-gauge",
        "faviconUrl": "static/brand/marvis-favicon.png?v=20260624-gauge",
        "validatorAliases": {},
        "source": "default",
    }


def test_branding_reads_validator_aliases_from_workspace_config(tmp_path: Path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "brand.json").write_text(
        json.dumps(
            {
                "platform_name": "本地平台",
                "validator_aliases": {
                    "  张三  ": "  小三  ",
                    "李四": "老四",
                    "bad-empty": "",
                    "non-string": 123,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = _client(tmp_path)

    payload = client.get("/api/branding").json()

    # Aliases are trimmed; empty / non-string entries are dropped.
    assert payload["validatorAliases"] == {"张三": "小三", "李四": "老四"}


def test_branding_validator_aliases_default_empty_for_missing_or_invalid_config(tmp_path: Path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "brand.json").write_text(
        json.dumps({"platform_name": "本地平台", "validator_aliases": ["not", "a", "map"]}),
        encoding="utf-8",
    )
    client = _client(tmp_path)

    payload = client.get("/api/branding").json()

    assert payload["validatorAliases"] == {}


def test_branding_reads_workspace_config_and_serves_asset(tmp_path: Path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "private-logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" aria-label="本地品牌徽标"></svg>',
        encoding="utf-8",
    )
    (branding_dir / "brand.json").write_text(
        json.dumps(
            {
                "platform_name": "本地风控模型验证平台",
                "browser_title": "本地智能模型验证平台",
                "primary_color": "#1f6feb",
                "logo": "private-logo.svg",
                "favicon": "private-logo.svg",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = _client(tmp_path)

    response = client.get("/api/branding")

    assert response.status_code == 200
    payload = response.json()
    assert payload["platformName"] == "本地风控模型验证平台"
    assert payload["browserTitle"] == "本地智能模型验证平台"
    assert payload["primaryColor"] == "#1f6feb"
    assert payload["logoUrl"].startswith("branding/assets/private-logo.svg?v=")
    assert payload["workspaceLogoUrl"].startswith("branding/assets/private-logo.svg?v=")
    assert payload["faviconUrl"].startswith("branding/assets/private-logo.svg?v=")
    assert payload["source"] == "workspace"

    asset_response = client.get(payload["logoUrl"])
    assert asset_response.status_code == 200
    assert 'aria-label="本地品牌徽标"' in asset_response.text


def test_index_html_is_prebranded_from_workspace_config(tmp_path: Path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "private-logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" aria-label="本地品牌徽标"></svg>',
        encoding="utf-8",
    )
    (branding_dir / "brand.json").write_text(
        json.dumps(
            {
                "platform_name": "本地风控模型验证平台",
                "browser_title": "本地智能模型验证平台",
                "primary_color": "#1f6feb",
                "logo": "private-logo.svg",
                "favicon": "private-logo.svg",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = _client(tmp_path)

    branding = client.get("/api/branding").json()
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "<title>本地智能模型验证平台</title>" in html
    assert f'href="{branding["faviconUrl"]}"' in html
    assert 'id="brandFaviconDark"' in html
    assert 'id="brandAppleTouchIconDark"' in html
    assert "static/brand/marvis-favicon-dark.png" not in html
    assert "static/brand/marvis-apple-touch-icon-dark.png" not in html
    assert '<meta name="apple-mobile-web-app-title" content="本地风控模型验证平台" />' in html
    assert f'src="{branding["logoUrl"]}"' in html
    assert f'src="{branding["workspaceLogoUrl"]}"' in html
    assert 'alt="本地风控模型验证平台 logo"' in html
    assert '<h1 id="platformName">本地风控模型验证平台</h1>' in html
    assert 'style="--brand-primary: #1f6feb;' in html
    assert "--brand-primary-hover:" in html
    assert "<title>MARVIS-全能风控智能体</title>" not in html


def test_branding_ignores_unsafe_asset_paths(tmp_path: Path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (tmp_path / "secret.svg").write_text("<svg>secret</svg>", encoding="utf-8")
    (branding_dir / "brand.json").write_text(
        json.dumps(
            {
                "platform_name": "Private",
                "browser_title": "Private",
                "primary_color": "#12zzzz",
                "logo": "../secret.svg",
                "favicon": "/etc/passwd",
            },
        ),
        encoding="utf-8",
    )
    client = _client(tmp_path)

    response = client.get("/api/branding")

    assert response.status_code == 200
    assert response.json() == {
        "platformName": "Private",
        "browserTitle": "Private",
        "primaryColor": "#303034",
        "logoUrl": "static/brand/marvis-workspace-logo.png?v=20260624-gauge",
        "workspaceLogoUrl": "static/brand/marvis-workspace-logo.png?v=20260624-gauge",
        "faviconUrl": "static/brand/marvis-favicon.png?v=20260624-gauge",
        "validatorAliases": {},
        "source": "workspace",
    }
    assert client.get("/branding/assets/../secret.svg").status_code == 404

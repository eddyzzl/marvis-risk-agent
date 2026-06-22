# Branding Configuration

Branding is runtime-configurable from the local workspace. Source code defaults are the public MARVIS brand, so the repository can be published without a separate OSS branch.

## Public Default

When no local config exists, the app uses:

- platform name: `MARVIS-全能信贷风控智能体`
- browser title: `MARVIS-全能信贷风控智能体`
- primary color: `#000000`
- logo: `marvis/static/brand/marvis-logo.png`
- favicon: `marvis/static/brand/marvis-favicon.png`

## Local Private Config

Create this ignored file:

```text
workspace/branding/brand.json
```

Example:

```json
{
  "platform_name": "本地信贷风控智能体",
  "browser_title": "本地信贷风控工作台",
  "primary_color": "#1f6feb",
  "logo": "private-logo.svg",
  "favicon": "private-logo.svg",
  "validator_aliases": {
    "张三": "小三",
    "李四": "老四"
  }
}
```

`validator_aliases` maps a real validator name to the display alias shown as the
agent's name. It is optional and lives only in this private config, so real names
never ship in the public bundle. Entries are trimmed; empty or non-string values
are ignored. When unset, the agent simply shows `Agent`.

Put referenced files next to the config, for example:

```text
workspace/branding/private-logo.svg
```

The app exposes the active brand at `GET /api/branding` and serves local assets through `/branding/assets/...`.

`GET /` also injects the active workspace brand into the initial HTML response for the first paint: browser title, favicon, sidebar logo, welcome logo, platform name, and primary color tokens. The frontend still calls `GET /api/branding` after load as a runtime refresh/fallback, but the first visible frame should already match the active local config.

Before publishing to GitHub, delete or omit `workspace/branding/`. The app should then fall back to the MARVIS public default.

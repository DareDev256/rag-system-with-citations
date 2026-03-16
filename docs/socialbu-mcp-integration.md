# SocialBu MCP Server Integration

> How any MCP-compatible AI agent can connect to SocialBu for automated social media management.

## What Is SocialBu?

[SocialBu](https://socialbu.com/) is an AI-powered social media management platform supporting **12 platforms**: Facebook, Instagram, X (Twitter), Threads, LinkedIn, TikTok, YouTube, Reddit, Mastodon, Pinterest, Bluesky, and Google Business Profile.

## MCP Integration Architecture

SocialBu does **not** publish a dedicated MCP server package. Instead, it exposes a full [OpenAPI v3.1 spec](https://socialbu.com/openapi.yaml) that generic OpenAPI-to-MCP proxy servers can consume — giving any MCP-compatible agent (Claude Desktop, Cursor, custom agents) access to the entire SocialBu API as native MCP tools.

```
┌──────────────┐     MCP Protocol     ┌──────────────────┐    HTTPS/REST    ┌──────────────┐
│  AI Agent    │◄────────────────────►│  OpenAPI-to-MCP  │◄───────────────►│  SocialBu    │
│  (Claude,    │     stdio/SSE        │  Proxy Server    │    Bearer JWT   │  API v1      │
│   Cursor)    │                      │                  │                 │              │
└──────────────┘                      └──────────────────┘                 └──────────────┘
```

## Setup Options

### Option A: Python Proxy (`mcp-openapi-proxy`)

**Source:** [github.com/matthewhand/mcp-openapi-proxy](https://github.com/matthewhand/mcp-openapi-proxy)

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "socialbu": {
      "command": "uvx",
      "args": ["mcp-openapi-proxy"],
      "env": {
        "OPENAPI_SPEC_URL": "https://socialbu.com/openapi.yaml",
        "API_KEY": "<your-socialbu-jwt-token>",
        "API_AUTH_TYPE": "Bearer"
      }
    }
  }
}
```

### Option B: Node Proxy (`openapi-mcp-server`)

**Source:** [github.com/janwilmake/openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server)

```json
{
  "mcpServers": {
    "socialbu": {
      "command": "npx",
      "args": ["openapi-mcp-server", "https://socialbu.com/openapi.yaml"],
      "env": {
        "API_KEY": "<your-socialbu-jwt-token>"
      }
    }
  }
}
```

## Authentication

1. Get credentials from your [SocialBu developer settings](https://socialbu.com/developers/docs)
2. Request a JWT: `POST https://socialbu.com/api/v1/auth/get_token`
3. Set the returned token as `API_KEY` in your MCP server config
4. The proxy injects `Authorization: Bearer <token>` on every request automatically

## Exposed MCP Tools

Once connected, the proxy registers every SocialBu API endpoint as an MCP tool:

| Category | Operations | Use Case |
|----------|-----------|----------|
| **Posts** | Create, read, update, delete, schedule | Schedule content across 12 platforms |
| **Queues** | List, add posts, shuffle order | Manage posting schedules |
| **Media** | Upload via pre-signed URL or public URL | Attach images/video (up to 500 MB) |
| **Accounts** | Connect, disconnect, update | Manage social platform connections |
| **AI Tools** | List available tools, execute by slug | Generate captions, post copy |
| **Insights** | Post counts, metrics, top posts, account metrics | Pull analytics and reporting |
| **Teams** | Create, update, delete teams | Manage team access and roles |
| **Curation** | Discover content | Content discovery and resharing |

## API Reference

| Detail | Value |
|--------|-------|
| Base URL | `https://socialbu.com/api/v1` |
| OpenAPI Spec | `https://socialbu.com/openapi.yaml` |
| Auth | Bearer JWT |
| Timestamps | UTC (`Y-m-d H:i:s`) |
| Pagination | `currentPage`, `lastPage`, `nextPage`, `total` |
| Media formats | jpg, png, gif, webp, mp4, mov, avi, webm, mkv, pdf |
| Max upload | 500 MB |

## Example Agent Workflow

A typical automated posting pipeline:

```python
# Pseudocode — what the agent does behind the scenes via MCP tools

# 1. Generate content using SocialBu's built-in AI
ai_tools = socialbu.list_ai_tools()
caption = socialbu.run_ai_tool("instagram-caption", topic="RAG systems")

# 2. Upload media
media = socialbu.upload_media_by_url("https://example.com/diagram.png")

# 3. Schedule the post across platforms
socialbu.create_post(
    content=caption,
    media_ids=[media["id"]],
    account_ids=["ig_account", "linkedin_account"],
    scheduled_at="2026-03-17 14:00:00"  # UTC
)

# 4. Check performance later
metrics = socialbu.get_post_metrics(post_id="abc123")
```

## Relevance to This Project

This RAG system generates citation-backed answers from document stores. Combined with SocialBu via MCP, an agent could:

- **Auto-publish research summaries** — query the RAG API, format the answer with citations, post to LinkedIn/X
- **Content pipeline automation** — ingest documents → generate grounded insights → schedule social posts
- **Analytics feedback loop** — pull SocialBu engagement metrics back into the RAG system as evaluation signals

## References

- [SocialBu Developer Docs](https://socialbu.com/developers/docs)
- [SocialBu API Help](https://help.socialbu.com/collections/1665992-apis)
- [OpenAPI Spec](https://socialbu.com/openapi.yaml)
- [mcp-openapi-proxy (PyPI)](https://pypi.org/project/mcp-openapi-proxy/)
- [openapi-mcp-server (GitHub)](https://github.com/janwilmake/openapi-mcp-server)

## What we are building
<!-- One paragraph: what, who uses it, what problem it solves -->

## Stack
- Language:
- Framework:
- Database:
- LLM provider:
- Cache:
- Package manager:
- Test runner:
- Other:

## Architecture
<!--
Describe layers and what each is responsible for.
Format: Layer name — what it does, what it owns, what it must not do

Example:
Gateway     — auth, rate limiting, routing. No business logic.
Core        — domain logic only. No DB calls directly, goes through lib/.
Inference   — all LLM calls. Single client. Streaming handled here only.
Storage     — DB + cache. No business logic. Returns typed models only.
-->

## Component responsibilities
<!--
One line per file/module describing exactly what it owns.
If two components share responsibility for something, that is a bug in the design.

Example:
lib/llm_client.py     — single LLM client instance, all completions go here
lib/context_builder.py — assembles final prompt, owns token budget logic
lib/rate_limiter.py   — Redis token bucket, called as dependency in routers
-->

## Data models
<!--
Key entities and their fields. Not full schema — just enough to code against.

Example:
User        — id, email, hashed_password, created_at
Conversation — id, user_id, created_at
Message     — id, conversation_id, role, content, tokens, created_at
-->

## API surface
<!--
Endpoints, method, auth required, what it does.

Example:
POST /auth/login         — public   — returns JWT
POST /chat               — JWT      — SSE streaming chat response
GET  /conversations      — JWT      — list user conversations
-->

## Implementation order
<!--
Ordered list. Each item is one focused implementer session.
Mark independent tasks [PARALLEL].

1. Scaffold — folder structure, requirements.txt, .env.example
2. Singletons — db.py, llm_client.py, redis_client.py
3. Models — schema.py data models
4. Auth — JWT sign/verify, /auth routes
5. ...
-->

## Environment variables
<!--
Every var the system needs. Copied to CLAUDE.md after scaffold.

VAR_NAME          — purpose, example value
-->

## Architecture decisions log
<!-- Append only. Never delete. Format: date: decision — reason -->

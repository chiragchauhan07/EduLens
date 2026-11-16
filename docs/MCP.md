# EduLens Model Context Protocol (MCP) Integration Specification

> **[ARCHIVED SPECIFICATION]:** In accordance with the final single-purpose production architecture of EduLens, the active application runtime has been simplified to focus exclusively on local student understanding and expressed confidence analysis. MCP integration code has been decoupled and pruned from the production runtime. This document is retained for historical and architectural reference.

## 1. Executive Summary & Overview

EduLens is a **Student Learning Intelligence and Performance Analytics Platform**. It is intentionally **not** an AI tutor. Instead, EduLens acts as an independent learning analytics engine that receives, stores, and evaluates student-AI conversations conducted on external foundation models (e.g., ChatGPT, Claude Desktop, Google Gemini).

The **Model Context Protocol (MCP)**, open-sourced by Anthropic and adopted across the AI industry, provides an open standard for connecting AI models to external tools, context, and storage via JSON-RPC 2.0. EduLens implements an MCP Server (`mcp>=2.0.0`) to serve as the unified bridge between students conversing with external AI agents and the EduLens machine learning analytics engine.

### Why EduLens Uses MCP
1. **Zero Client-Side Burden:** Students interact naturally with their preferred AI agents (ChatGPT, Claude, etc.) without having to copy-paste chat logs or install browser extensions.
2. **Standardized Context & Tool Calling:** AI models invoke structured MCP tools during or at the end of learning sessions to preserve conversations and retrieve pedagogical feedback.
3. **Multi-User Security & Isolation:** MCP headers propagate bearer tokens that bind every ingested session, quiz submission, and analytics query strictly to the authenticated student.
4. **Bi-Directional Learning Intelligence:** Beyond ingesting conversations, MCP allows AI agents to query the student's historical topic performance and mastery levels in real time to adapt their explanations.

---

## 2. System Architecture

```
+---------------------------------------------------------------------------------------------+
|                                    EXTERNAL AI AGENTS                                       |
|                                                                                             |
|   +--------------------------+                         +--------------------------------+   |
|   |         ChatGPT          |                         |         Claude Desktop         |   |
|   | (Streamable HTTP / SSE)  |                         |  (Local Stdio or Remote HTTP)  |   |
|   +------------+-------------+                         +---------------+----------------+   |
+----------------|-------------------------------------------------------|--------------------+
                 | Authorization: Bearer <token>                         | JSON-RPC 2.0 (stdio / HTTP)
                 v                                                       v
+---------------------------------------------------------------------------------------------+
|                                  EDULENS MCP SERVER LAYER                                   |
|                                                                                             |
|   +-------------------------------------------------------------------------------------+   |
|   |                   Authentication & Context Layer (TokenManager)                     |   |
|   |      - Validates Bearer tokens in Authorization headers                             |   |
|   |      - ContextVar-driven user session isolation (UserContext: student_id)           |   |
|   |      - Automatic context cleanup preventing cross-task leakage                      |   |
|   +------------------------------------------+------------------------------------------+   |
|                                              |                                              |
|   +------------------------------------------v------------------------------------------+   |
|   |                       Registered MCP Tools & Capabilities                           |   |
|   |                                                                                     |   |
|   |   1. ingest_learning_session     4. get_topic_analytics                             |   |
|   |   2. get_session_analytics       5. list_learning_sessions                          |   |
|   |   3. get_student_performance     6. submit_quiz_results                             |   |
|   |                                                                                     |   |
|   |   Resources: edulens://curriculum/topics  |  edulens://health                       |   |
|   |   Prompts:   save_learning_session                                                  |   |
|   +---------------------+------------------------------------+--------------------------+   |
+-------------------------|------------------------------------|------------------------------+
                          |                                    |
                          v Direct Service Call                v Thread-Safe Operations
+-----------------------------------------------+   +-----------------------------------------+
|        DSP / NLP / ML ANALYTICS ENGINE        |   |         PERSISTENT SQLITE STORAGE       |
|                                               |   |             (data/storage/edulens.db)   |
|   - Conversational Feature Extractor (8 dims) |   |                                         |
|   - TF-IDF Vectorizer (100 terms, unigrams)   |   |   - raw_sessions                        |
|   - Logistic Regression Classifier            |   |     (audit-proof raw conversation JSON) |
|   - Ridge Quiz Regressor                      |   |   - analytics_results                   |
|   - Topic Performance Scoring Engine          |   |     (derived ML predictions & scores)   |
|   - Local & Global Feature Explainability     |   |   - quiz_submissions                    |
+-----------------------------------------------+   |     (evaluated item-level quiz data)    |
                                                    +-----------------------------------------+
```

---

## 3. Supported Transports & Production Recommendations

EduLens supports all three standard MCP transports:

| Transport | CLI Option | Endpoint / Mode | Status & Recommendation | Protocol Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Streamable HTTP** *(Default)* | `--transport streamable` | `POST /mcp` | **Recommended for HTTP-based clients & MCP Inspector** | Bidirectional chunked HTTP POST streaming JSON-RPC messages with session headers (`Mcp-Session-Id`). |
| **Server-Sent Events (SSE)** | `--transport sse` | `GET /sse`, `POST /messages/` | Supported for legacy remote clients & debugging | Continuous SSE stream on GET `/sse` with separate HTTP POST endpoint for client requests. |
| **Standard I/O (Stdio)** | `--transport stdio` | `stdin` / `stdout` | **Verified for Claude Desktop (Local Machine)** | Fast subprocess communication using JSON-RPC lines over standard input/output. |

> [!TIP]
> **Streamable HTTP:** When deploying over HTTP, use `--transport streamable`. Streamable HTTP is stateless-friendly, multiplexes over standard HTTP connections, avoids long-lived connection exhaustion on cloud proxies, and is natively supported by modern MCP clients and inspectors at the standard `/mcp` path.

---

## 4. MCP Tools Reference & Schema Specification

The EduLens MCP server registers six core tools. All parameter names, types, and return payloads correspond directly to `backend/mcp_server/server.py` and `backend/mcp_server/schemas/session_schema.py`.

> [!IMPORTANT]
> **Authentication Parameter (`auth_token`) vs. HTTP Middleware:**  
> When HTTP authentication is active (`EDULENS_AUTH_ENABLED=true`), the server's `AuthMiddleware` verifies credentials at the HTTP boundary before request bodies are parsed or tools are dispatched. Unauthenticated HTTP requests receive an immediate `401 Unauthorized`.  
> Therefore, **the `auth_token` tool argument cannot bypass missing HTTP headers.**  
> `auth_token` is an optional fallback designed strictly for:
> 1. Local `stdio` transport where HTTP headers do not exist.
> 2. Embedded Python service calls invoking tool functions directly.
> 3. Environments where HTTP authentication is disabled (`--no-auth`), but per-call session tokens are passed.

---

### Tool 1: `ingest_learning_session`
Ingests a complete student-AI conversation, saves the raw transcript to persistent storage before execution, executes the 4-dimension ML pipeline, and returns real-time learning analytics.

- **Parameters:**
  - `messages` (`array[object]`, **required**): Chronological list of message objects:
    - `role` (`string`, **required**): Speaker role. Allowed values: `user`, `student`, `assistant`, `teacher`, `system`, `human`, `ai`. At least one student message (`user`, `student`, or `human`) is required.
    - `content` (`string`, **required**): Non-empty text content of the utterance.
    - `timestamp` (`string`, optional): ISO-8601 formatted timestamp.
  - `subject` (`string`, optional): Academic subject (e.g., `"Computer Science"`, `"Algorithms"`).
  - `topic` (`string`, optional): Specific curriculum topic (e.g., `"Binary Search"`, `"Dynamic Programming"`).
  - `session_id` (`string`, optional): Custom unique session ID (auto-generated if omitted).
  - `student_id` (`string`, optional): Student identifier. If provided, must match the authenticated identity.
  - `source_agent` (`string`, optional, default: `"chatgpt"`): Originating AI platform (`"chatgpt"`, `"claude"`, `"generic"`).
  - `metadata` (`object`, optional): Supplementary session metadata (e.g., duration, client version).
  - `quiz_submission` (`object`, optional): Pre-completed quiz questions and student answers for deterministic grading.
  - `auth_token` (`string`, optional): Fallback authentication token (for stdio/embedded use).

- **Input JSON Schema:**
```json
{
  "type": "object",
  "properties": {
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": {"type": "string", "enum": ["user", "student", "assistant", "teacher", "system", "human", "ai"]},
          "content": {"type": "string"},
          "timestamp": {"type": "string"}
        },
        "required": ["role", "content"]
      }
    },
    "subject": {"type": "string"},
    "topic": {"type": "string"},
    "session_id": {"type": "string"},
    "student_id": {"type": "string"},
    "source_agent": {"type": "string", "default": "chatgpt"},
    "metadata": {"type": "object"},
    "quiz_submission": {"type": "object"},
    "auth_token": {"type": "string"}
  },
  "required": ["messages"]
}
```

- **Output Example:**
```json
{
  "status": "success",
  "session_id": "SES_1E28D28D503C",
  "user_id": "student_001",
  "source_agent": "chatgpt",
  "subject": "Computer Science",
  "topic": "Binary Search",
  "ingested_at": "2026-11-15T23:30:00Z",
  "turns_count": 3,
  "is_duplicate": false,
  "message": "Learning conversation successfully ingested and evaluated across 4 dimensions.",
  "analytics": {
    "status": "success",
    "topics": [
      {
        "topic": "Binary Search",
        "understanding": {
          "classification": "understood",
          "score_index": 90.0,
          "confidence": 0.958,
          "probabilities": {
            "understood": 0.958,
            "partial": 0.032,
            "confused": 0.010
          },
          "display_name": "Understood (Demonstrated Conceptual Clarity)",
          "pedagogical_feedback": "Student demonstrated solid understanding. Ready for advanced variations."
        },
        "actual_quiz_performance": {
          "status": "awaiting_quiz_submission"
        },
        "expressed_confidence": {
          "classification": "high",
          "score": 85.0,
          "status": "experimental_heuristic_baseline"
        },
        "overall_performance": {
          "score": 88.5,
          "status": "awaiting_quiz_submission",
          "formula": "(90.0 * 0.70) + (85.0 * 0.30) = 88.5/100",
          "explanation": "Calculated using available conversational dimensions awaiting quiz.",
          "discrepancies": []
        },
        "conversational_signals": {
          "num_turns": 2.0,
          "total_words": 18.0,
          "avg_turn_length": 9.0,
          "question_count": 1.0,
          "confusion_markers": 0.0,
          "clarification_markers": 0.0,
          "understanding_markers": 1.0,
          "code_indicators": 0.0
        },
        "feature_explanation": {
          "predicted_class": "understood",
          "top_positive_features": [
            {"feature": "signal_understanding_markers", "contribution": 2.14},
            {"feature": "signal_total_words", "contribution": 0.85}
          ]
        }
      }
    ]
  }
}
```

---

### Tool 2: `get_session_analytics`
Retrieves stored multi-dimensional learning analytics for a previously ingested session.

- **Parameters:**
  - `session_id` (`string`, **required**): Unique session identifier returned during ingestion.
  - `auth_token` (`string`, optional): Fallback authentication token.

- **Output Example:**
```json
{
  "session_id": "SES_1E28D28D503C",
  "user_id": "student_001",
  "subject": "Computer Science",
  "topic": "Binary Search",
  "source_agent": "chatgpt",
  "status": "processed",
  "topic_understanding": {
    "predicted_class": "understood",
    "understanding_score": 90.0,
    "confidence": 0.958,
    "display_name": "Understood"
  },
  "actual_quiz_performance": {
    "status": "awaiting_quiz_submission"
  },
  "learning_confidence": {
    "expressed_confidence_level": "high",
    "expressed_confidence_score": 85.0,
    "status": "experimental_heuristic_baseline"
  },
  "overall_topic_performance": {
    "overall_performance_score": 88.5,
    "status": "awaiting_quiz_submission"
  },
  "discrepancy_analysis": {
    "discrepancy_detected": false,
    "details": null
  },
  "pedagogical_feedback": "Student demonstrated solid understanding.",
  "processed_at": "2026-11-15T23:30:00Z"
}
```

---

### Tool 3: `get_student_performance`
Retrieves the student's accumulated longitudinal performance portfolio across all historical sessions.

- **Parameters:**
  - `student_id` (`string`, optional): Student identifier. Defaults to the authenticated user. If a different `student_id` is passed, the server enforces strict isolation and raises an `AuthorizationError`.
  - `auth_token` (`string`, optional): Fallback authentication token.

- **Output Example:**
```json
{
  "student_id": "student_001",
  "total_sessions": 14,
  "topics_studied": ["Binary Search", "Quicksort", "Recursion", "Trees"],
  "understanding_distribution": {
    "understood": 10,
    "partial": 3,
    "confused": 1
  },
  "average_understanding_score": 82.5,
  "average_actual_quiz_score": 88.0,
  "average_confidence_score": 76.4,
  "average_overall_score": 84.1,
  "topic_wise_breakdown": [
    {
      "topic": "Binary Search",
      "subject": "Computer Science",
      "sessions": 4,
      "mastery_rate": 100.0,
      "latest_understanding": "understood"
    }
  ],
  "areas_of_confusion": ["Dynamic Programming"],
  "learning_progress": []
}
```

---

### Tool 4: `get_topic_analytics`
Retrieves curriculum-level performance metrics for a specific topic across the student's history.

- **Parameters:**
  - `topic` (`string`, **required**): Name of curriculum topic.
  - `student_id` (`string`, optional): Student identifier.
  - `auth_token` (`string`, optional): Fallback authentication token.

- **Output Example:**
```json
{
  "topic": "Binary Search",
  "subject": "Computer Science",
  "student_id": "student_001",
  "total_sessions": 4,
  "understanding_distribution": {"understood": 4, "partial": 0, "confused": 0},
  "average_understanding_score": 90.0,
  "quiz_performance": {
    "average_score_percent": 95.0,
    "quizzes_taken": 2
  },
  "expressed_confidence": {
    "average_confidence_score": 88.0
  },
  "overall_topic_performance": {
    "composite_score": 91.5
  },
  "historical_trend": []
}
```

---

### Tool 5: `list_learning_sessions`
Lists historical learning sessions belonging to the student with filtering and pagination.

- **Parameters:**
  - `subject` (`string`, optional): Subject filter.
  - `topic` (`string`, optional): Topic filter.
  - `date_from` (`string`, optional): ISO-8601 start date filter.
  - `date_to` (`string`, optional): ISO-8601 end date filter.
  - `limit` (`integer`, optional, default: 20): Maximum sessions to return (1-100).
  - `offset` (`integer`, optional, default: 0): Pagination offset (min: 0).
  - `auth_token` (`string`, optional): Fallback authentication token.

- **Output Example:**
```json
{
  "total_count": 1,
  "limit": 20,
  "offset": 0,
  "sessions": [
    {
      "session_id": "SES_1E28D28D503C",
      "user_id": "student_001",
      "subject": "Computer Science",
      "topic": "Binary Search",
      "source_agent": "chatgpt",
      "status": "processed",
      "understanding_class": "understood",
      "overall_score": 88.5,
      "ingested_at": "2026-11-15T23:30:00Z"
    }
  ]
}
```

---

### Tool 6: `submit_quiz_results`
Evaluates post-session assessment questions deterministically against answer keys and updates composite topic mastery.

- **Parameters:**
  - `questions` (`array[object]`, **required**): List of question items with question identifiers, student answers, and correct answers (or choices).
  - `session_id` (`string`, optional): Associated session ID to link and update.
  - `subject` (`string`, optional): Academic subject.
  - `topic` (`string`, optional): Assessed topic.
  - `answer_key` (`object`, optional): Answer key mapping.
  - `student_answers` (`object`, optional): Student response mapping.
  - `auth_token` (`string`, optional): Fallback authentication token.

- **Output Example:**
```json
{
  "quiz_id": "QZ_DDC1FD8BD214",
  "session_id": "SES_1E28D28D503C",
  "user_id": "student_001",
  "total_questions": 2,
  "correct_count": 2,
  "incorrect_count": 0,
  "unanswered_count": 0,
  "actual_score_percent": 100.0,
  "question_breakdown": [],
  "topic_breakdown": {"General": {"score_percent": 100.0}},
  "evaluated_at": "2026-11-15T23:35:00Z",
  "updated_overall_performance": {
    "topic": "Binary Search",
    "overall_performance_score": 92.0,
    "status": "complete",
    "calculation_formula": "(90.0 * 0.40) + (100.0 * 0.40) + (85.0 * 0.20) = 92.0/100"
  }
}
```

---

## 5. Storage Architecture & Schema

Persistent storage separates audit-proof raw logs from computed analytics using SQLite in WAL mode (`data/storage/edulens.db`).

```sql
-- Raw Conversational Records (Audit-Proof)
CREATE TABLE IF NOT EXISTS raw_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    subject TEXT,
    topic TEXT,
    content_hash TEXT NOT NULL,
    turns_count INTEGER NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT,
    ingested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_sessions_user ON raw_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_raw_sessions_hash ON raw_sessions(user_id, content_hash);

-- Derived Machine Learning Analytics
CREATE TABLE IF NOT EXISTS analytics_results (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    understanding_score REAL NOT NULL,
    confidence REAL NOT NULL,
    actual_quiz_score REAL,
    quiz_status TEXT NOT NULL,
    expressed_confidence_level TEXT NOT NULL,
    expressed_confidence_score REAL NOT NULL,
    overall_performance_score REAL,
    performance_category TEXT,
    discrepancy_detected INTEGER NOT NULL,
    discrepancy_details TEXT,
    pedagogical_feedback TEXT NOT NULL,
    full_analytics_json TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES raw_sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics_results(user_id);

-- Quiz Submissions & Performance
CREATE TABLE IF NOT EXISTS quiz_submissions (
    quiz_id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT NOT NULL,
    subject TEXT,
    topic TEXT,
    total_questions INTEGER NOT NULL,
    correct_count INTEGER NOT NULL,
    incorrect_count INTEGER NOT NULL,
    unanswered_count INTEGER NOT NULL,
    actual_score_percent REAL NOT NULL,
    question_breakdown_json TEXT NOT NULL,
    topic_breakdown_json TEXT NOT NULL,
    raw_submission_json TEXT NOT NULL,
    evaluated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiz_user ON quiz_submissions(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_session ON quiz_submissions(session_id);
```

---

## 6. Authentication Compatibility & Limitations Analysis

### Current Implementation: In-Memory TokenManager & Static Bearer Validation

The current EduLens MCP implementation uses an in-memory `TokenManager` that resolves API tokens from an in-memory dictionary or the `EDULENS_API_TOKENS` environment variable. The ASGI `AuthMiddleware` verifies that incoming HTTP requests carry an `Authorization: Bearer <token>` header.

#### What Works Today:
1. **Local Development & Automated Tests:**
   - Pre-configured development tokens (`EDULENS_DEV_TOKEN`) allow zero-friction local testing.
   - The test suite verifies full authentication flows and multi-user isolation.
2. **Authenticated MCP Clients Supporting Custom HTTP Headers:**
   - **Claude Desktop:** Supports adding custom headers (`Authorization: Bearer <token>`) in `claude_desktop_config.json` for remote HTTP connections, or local environment variables for stdio.
   - **Official MCP Inspector:** Allows specifying custom HTTP headers (`Authorization: Bearer <token>`) when connecting to `http://127.0.0.1:8001/mcp`.
   - **Custom SDK / CLI Scripts:** Any HTTP client capable of attaching an `Authorization` header can successfully authenticate.

#### What Does NOT Work with ChatGPT:
- **ChatGPT Custom MCP Connection Authentication:**
  - OpenAI's custom MCP server implementation in ChatGPT does **not** allow users to manually type a static Bearer token or custom HTTP header in the connection UI.
  - ChatGPT custom MCP connections support either **No Auth** (public unauthenticated servers) or **OAuth 2.1** (Authorization Code flow with PKCE).
  - Consequently, **the current EduLens server cannot authenticate a live ChatGPT custom MCP connection** when authentication is enforced. Static Bearer token authentication is **not** equivalent to OAuth 2.1.

### Requirements for ChatGPT OAuth 2.1 Compatibility

To achieve full authentication compatibility with ChatGPT Web, EduLens would require a dedicated OAuth 2.1 Authorization Server layer comprising:

1. **Authorization Endpoint (`/oauth/authorize`):**
   - Implements RFC 7636 Proof Key for Code Exchange (PKCE) with `code_challenge` and `code_challenge_method=S256`.
   - Renders a user login and consent UI where the student confirms granting ChatGPT permission to ingest sessions and query learning analytics.
   - Issues short-lived, single-use authorization codes.
2. **Token Exchange Endpoint (`/oauth/token`):**
   - Validates authorization codes against client credentials and the cryptographic `code_verifier`.
   - Issues cryptographically signed Access Tokens (RS256 JWTs) containing `sub` (student ID) and scoped permissions.
3. **OAuth 2.1 Discovery Metadata (`/.well-known/oauth-authorization-server` or `/.well-known/openid-configuration`):**
   - Exposes standardized endpoints and supported grant types (`authorization_code`) for automatic discovery by ChatGPT.
4. **Client Registration & Callback Management:**
   - Pre-registers OpenAI client credentials (`client_id`, `client_secret`) and authorized redirect URIs.
5. **Persistent Session & Token Revocation Store:**
   - PostgreSQL or Redis store tracking issued tokens, expiration, and real-time revocation (replacing the ephemeral in-memory dictionary).

### Architecture Roadmap for OAuth 2.1 Implementation

Instead of introducing an incomplete OAuth implementation during this cleanup, the following phased architecture plan will be executed in a dedicated security release:

```
Step 1: Database Schema  -> Define oauth_clients, oauth_codes, and oauth_tokens tables in SQLite/PostgreSQL.
Step 2: PKCE Engine      -> Build code_challenge / code_verifier verification module using hashlib SHA-256.
Step 3: Endpoints        -> Expose /oauth/authorize (with consent template) and /oauth/token (returning JWTs).
Step 4: Discovery        -> Add /.well-known/oauth-authorization-server route.
Step 5: E2E Validation   -> Test the flow using an external OAuth 2.1 compliance suite before linking to ChatGPT.
```

---

## 7. Client Setup Guides

### A. ChatGPT Custom MCP Server Integration

OpenAI allows connecting custom MCP servers in ChatGPT under specific subscription and governance tiers according to [official OpenAI documentation](https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt):

#### Official Plan & Capability Matrix:
- **ChatGPT Business and Enterprise / Edu:** **Full MCP Access.** Supports both read/fetch and action/write tool invocations.
- **ChatGPT Pro:** **Read/Fetch Limitations.** Restricted to information retrieval; action tools are gated or restricted.
- **ChatGPT Plus & ChatGPT Go:** **Not Officially Supported.** Custom MCP server connections are not officially offered for Plus or Go tiers. Do not attempt custom MCP setup on these tiers without official platform updates.

#### Configuration Workflow in ChatGPT:
1. Open ChatGPT (Web or Desktop) under an eligible Enterprise, Business, or Pro account.
2. Navigate to **Workspace Settings** -> **Plugins & Integrations** (or **Settings** -> **Connected Apps / MCP Servers**).
3. Select **Add Custom MCP Server**.
4. Configure connection parameters:
   - **Server URL:** `https://<your-edulens-domain>/mcp`
   - **Transport:** Streamable HTTP
   - **Authentication:**  
     - *Public / Development Mode:* If running with `--no-auth` (unauthenticated).
     - *OAuth 2.1 Mode:* Requires the OAuth 2.1 Authorization Server described in Section 6. Static Bearer tokens cannot be entered manually in this UI.
5. Save connection. ChatGPT discovers available tools and registers them for eligible sessions.

#### Native MCP vs. GPT Actions:
- **Native MCP (`/mcp`):** Standardized JSON-RPC 2.0 streaming protocol over Streamable HTTP or SSE, supporting dynamic discovery of tools, resources, and prompts.
- **GPT Actions (`/api/pipeline`):** Legacy OpenAPI 3.0 REST endpoints configured via the Custom GPT builder. EduLens provides backward-compatible REST endpoints in `api/index.py` for GPT Actions.

---

### B. Claude Desktop Integration

Claude Desktop supports both local Standard I/O (Stdio) and remote HTTP/SSE transports.

#### 1. Local Stdio Transport (Verified for Local Machine)
Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "edulens": {
      "command": "<path-to-your-venv>/Scripts/python.exe",
      "args": [
        "-m",
        "backend.mcp_server.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "EDULENS_AUTH_ENABLED": "true",
        "EDULENS_DEV_TOKEN": "<your-secure-token>"
      }
    }
  }
}
```

#### 2. Remote Streamable HTTP Transport
```json
{
  "mcpServers": {
    "edulens-remote": {
      "url": "http://127.0.0.1:8001/mcp",
      "headers": {
        "Authorization": "Bearer <your-secure-token>"
      }
    }
  }
}
```

---

### C. Testing with Official MCP Inspector

The official Model Context Protocol Inspector connects over Streamable HTTP (`/mcp`):

```powershell
# Launch MCP Inspector pointing to EduLens Streamable HTTP endpoint
npx @modelcontextprotocol/inspector http://127.0.0.1:8001/mcp
```

Inside the Inspector UI:
1. Add Header: `Authorization: Bearer <your-secure-token>`.
2. Click **Connect**.
3. View and execute all 6 tools, inspect resources (`edulens://curriculum/topics`), and review JSON-RPC logs.

---

## 8. Verification Status & Test Results

### Current Verification Level:
- **Automated Test Suite:** **78/78 Tests Passing** in `pytest` (ML, API, DSP, Storage, Security, ContextVar isolation, Stdio, Streamable HTTP).
- **Official MCP Inspector:** **Verified** against `http://127.0.0.1:8001/mcp` across all 6 tools, error responses, and user isolation.
- **Claude Desktop:** **Verified** on local stdio transport via subprocess `ClientSession`.
- **ChatGPT Custom MCP:** **Unverified in live production.** Requires implementation of the OAuth 2.1 Authorization Server before live enterprise testing can occur.

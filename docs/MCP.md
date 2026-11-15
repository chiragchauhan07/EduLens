# EduLens Model Context Protocol (MCP) Integration Specification

## 1. Executive Summary & Overview

EduLens is a **Student Learning Intelligence and Performance Analytics Platform**. It is intentionally **not** an AI tutor. Instead, EduLens acts as an independent learning analytics engine that receives, stores, and evaluates student-AI conversations conducted on external foundation models (e.g., ChatGPT, Claude Desktop, Google Gemini).

The **Model Context Protocol (MCP)**, open-sourced by Anthropic and adopted across the AI industry, provides an open standard for connecting AI models to external tools, context, and storage via JSON-RPC 2.0. EduLens implements an official MCP Server (`mcp>=2.0.0`) to serve as the unified bridge between students conversing with external AI agents and the EduLens machine learning analytics engine.

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
|   | (Streamable HTTP / SSE)  |                         |  (Local Stdio or Remote SSE)   |   |
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

| Transport | CLI Option | Endpoint / Mode | Recommended For | Protocol Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Streamable HTTP** *(Default & Recommended)* | `--transport streamable` | `POST /mcp` | **Production Deployments, Cloud Hosting & ChatGPT** | Bidirectional chunked HTTP POST streaming JSON-RPC messages with session headers (`Mcp-Session-Id`). |
| **Server-Sent Events (SSE)** | `--transport sse` | `GET /sse`, `POST /messages/` | Legacy remote clients & debugging | Continuous SSE stream on GET `/sse` with separate HTTP POST endpoint for client requests. |
| **Standard I/O (Stdio)** | `--transport stdio` | `stdin` / `stdout` | **Claude Desktop (Local Machine)** | Fast subprocess communication using JSON-RPC lines over standard input/output. |

> [!TIP]
> **Production Recommendation:** Always deploy with `--transport streamable`. Streamable HTTP is stateless-friendly, multiplexes over standard HTTP/1.1 and HTTP/2 connections, avoids sticky long-lived connection exhaustion on cloud proxies, and is natively supported by modern MCP clients and inspectors at the standard `/mcp` path.

---

## 4. MCP Tools Reference & Schema Specification

The EduLens MCP server registers six production tools. All parameter names, types, and return payloads correspond directly to `backend/mcp_server/server.py` and `backend/mcp_server/schemas/session_schema.py`.

### Tool 1: `ingest_learning_session`
Ingests a complete student-AI conversation, saves the raw conversation transcript to persistent storage before execution, executes the 4-dimension ML pipeline, and returns real-time learning analytics.

- **Parameters:**
  - `messages` (`array[object]`, required): Chronological list of message objects, each containing:
    - `role` (`string`, required): `"user"` / `"student"` for learner, `"assistant"` / `"teacher"` for AI.
    - `content` (`string`, required): Text content of the utterance.
    - `timestamp` (`string`, optional): ISO-8601 timestamp.
  - `subject` (`string`, optional): Subject area (e.g. `"Computer Science"`, `"Algorithms"`).
  - `topic` (`string`, optional): Curriculum topic (e.g. `"Binary Search"`, `"Dynamic Programming"`).
  - `session_id` (`string`, optional): Custom session ID (auto-generated if omitted).
  - `student_id` (`string`, optional): Student identifier. Must match authenticated identity.
  - `source_agent` (`string`, optional, default: `"chatgpt"`): Originating AI (`"chatgpt"`, `"claude"`).
  - `metadata` (`object`, optional): Supplementary session metadata.
  - `quiz_submission` (`object`, optional): Pre-completed quiz questions and student answers.
  - `auth_token` (`string`, optional): Authentication token if not provided in HTTP headers.

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
          "role": {"type": "string", "enum": ["user", "student", "assistant", "teacher", "system"]},
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

- **Output JSON Schema & Example:**
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
  - `session_id` (`string`, required): Identifier returned during ingestion.
  - `auth_token` (`string`, optional): Token override if not in HTTP headers.

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
Retrieves the student's accumulated longitudinal performance portfolio.

- **Parameters:**
  - `student_id` (`string`, optional): Validated against authenticated identity.
  - `auth_token` (`string`, optional): Auth token override.

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
  - `topic` (`string`, required): Name of topic.
  - `student_id` (`string`, optional): Target student.
  - `auth_token` (`string`, optional): Auth token override.

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
  - `date_from` (`string`, optional): ISO-8601 start date.
  - `date_to` (`string`, optional): ISO-8601 end date.
  - `limit` (`integer`, optional, default: 20): Max sessions to return (1-100).
  - `offset` (`integer`, optional, default: 0): Pagination offset.
  - `auth_token` (`string`, optional): Auth token override.

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
  - `questions` (`array[object]`, required): List of questions with `question_id`, `student_answer`, and `correct_answer`.
  - `session_id` (`string`, optional): Associated session ID to update.
  - `subject` (`string`, optional): Subject name.
  - `topic` (`string`, optional): Topic assessed.
  - `answer_key` (`object`, optional): Answer key mapping.
  - `student_answers` (`object`, optional): Student response mapping.
  - `auth_token` (`string`, optional): Auth token override.

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

## 6. Authentication Architecture & Production Roadmap

### Current Authentication & Limitations of In-Memory TokenManager

The current EduLens MCP implementation uses a lightweight, in-memory `TokenManager` for API token resolution. While effective for local development and integration tests, it has specific limitations:

1. **Server Restarts:** Tokens added at runtime in memory are reset on server restart (unless persisted or declared in `EDULENS_API_TOKENS`).
2. **Multi-Worker Scaling:** Multi-process uvicorn workers (`--workers 4`) do not share Python heap memory. A token registered dynamically in worker A will be missing in worker B.
3. **Token Revocation & Rotation:** Lacks distributed revocation lists and token expiry metadata.
4. **User Identity Persistence:** Does not store user credentials in a persistent database table.

### Recommended Production Authentication Architecture
For production cloud environments, EduLens should be configured with:
1. **Database-Backed Credentials:** A PostgreSQL `api_tokens` table with SHA-256 hashed keys and user ID bindings.
2. **Stateless JWTs (RS256):** Tokens containing cryptographic claims (`sub: student_id`, `exp`, `iss: edulens`) verifiable statelessly across all worker instances.
3. **Redis Session Store:** Fast distributed cache for instant token revocation and per-student rate limiting.
4. **OAuth 2.0 Authorization Server:** Implementing the standard OAuth 2.0 Authorization Code flow with PKCE (`/oauth/authorize`, `/oauth/token`) to provide 1-click workspace authorization in ChatGPT Web and third-party educational LMS platforms.

---

## 7. Client Setup Guides

### A. ChatGPT Custom MCP Server Integration

OpenAI supports connecting custom MCP servers directly to ChatGPT for users on **ChatGPT Plus, Team, Enterprise, and Edu** plans.

#### Connection Method:
1. Open ChatGPT (Web or Desktop).
2. Navigate to **Workspace Settings** -> **Connected Apps** (or **Developer Settings** -> **Custom MCP Servers**).
3. Click **Add Custom MCP Server**.
4. Configure connection parameters:
   - **Server URL:** `https://<your-edulens-domain>/mcp` (Streamable HTTP, recommended) or `/sse`.
   - **Transport:** Streamable HTTP.
   - **Authentication:**
     - *Custom Header (Developer / Preview Mode):* Add Header `Authorization` with value `Bearer <your-student-api-token>`.
     - *OAuth 2.0 (Enterprise Workspace Mode):* Note that if your ChatGPT workspace strictly mandates an OAuth 2.0 Authorization flow, an OAuth provider layer (e.g. Auth0 / FastAPI OAuth) must front the MCP server.
5. Save the connection. ChatGPT will automatically call `POST /mcp` to initialize, discover all 6 EduLens tools via `tools/list`, and enable them in your chat sessions.

#### Difference Between Native MCP and GPT Actions:
- **Native MCP (`/mcp`):** Uses JSON-RPC 2.0 over Streamable HTTP or SSE. The AI client dynamically negotiates tools, resources, and session contexts.
- **GPT Actions (`/api/pipeline`):** Uses traditional REST and OpenAPI 3.0 schemas. EduLens maintains backward compatibility with GPT Actions via `api/index.py` at `/api/pipeline`.

---

### B. Claude Desktop Integration

Claude Desktop connects seamlessly via either local Stdio or remote SSE.

#### 1. Local Stdio Transport (Recommended for Desktop)
Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "edulens": {
      "command": "c:/Users/LENOVO/OneDrive/Documents/EduLens/.venv/Scripts/python.exe",
      "args": [
        "-m",
        "backend.mcp_server.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "EDULENS_AUTH_ENABLED": "true",
        "EDULENS_DEV_TOKEN": "edulens-dev-token-student-001"
      }
    }
  }
}
```

#### 2. Remote Streamable HTTP / SSE Transport
```json
{
  "mcpServers": {
    "edulens-remote": {
      "url": "http://127.0.0.1:8001/mcp",
      "headers": {
        "Authorization": "Bearer edulens-dev-token-student-001"
      }
    }
  }
}
```

---

### C. Testing with Official MCP Inspector

You can test and inspect all tools using the official Model Context Protocol Inspector:

```bash
# Launch MCP Inspector pointing to EduLens Streamable HTTP endpoint
npx @modelcontextprotocol/inspector http://127.0.0.1:8001/mcp
```
Inside the Inspector UI:
1. Add Header: `Authorization: Bearer edulens-dev-token-student-001`.
2. Click **Connect**.
3. View and execute all 6 tools, inspect resources (`edulens://curriculum/topics`), and review JSON-RPC logs.

---

## 8. Automated Testing & Verification

The test suite validates all transport modes, tools, authentication, and isolation guarantees:

```bash
# Run complete test suite (78 passing tests)
.\.venv\Scripts\pytest tests/ -v

# Run MCP server tests specifically (36 tests)
.\.venv\Scripts\pytest tests/test_mcp_server.py -v
```

**Verification Results:**
- **36/36 MCP Tests Passing:** Validating all 6 tools, Streamable HTTP `/mcp`, SSE, Stdio, query parameter blocking, identity forgery prevention, and ContextVar task isolation.
- **1/1 Live E2E Subprocess Test Passing:** Official `ClientSession` interaction over Stdio.
- **78/78 Total Project Tests Passing** across ML, API, and MCP suites.

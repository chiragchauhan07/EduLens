# EduLens Model Context Protocol (MCP) Integration Specification

## 1. Executive Summary & Overview

EduLens is a **Student Learning Intelligence and Performance Analytics Platform**. It is intentionally **not** an AI tutor. Instead, EduLens acts as an independent learning analytics engine that receives, stores, and evaluates student-AI conversations conducted on external foundation models (e.g., ChatGPT, Claude Desktop, Google Gemini).

The **Model Context Protocol (MCP)**, open-sourced by Anthropic, provides an open standard for connecting AI models to external tools, context, and storage. EduLens implements an official MCP Server (`mcp>=2.0.0`) to serve as the unified bridge between students conversing with external AI agents and the EduLens machine learning analytics engine.

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
|   |  (Remote SSE / HTTP)     |                         |  (Local Stdio or Remote SSE)   |   |
|   +------------+-------------+                         +---------------+----------------+   |
+----------------|-------------------------------------------------------|--------------------+
                 | Authorization: Bearer <token>                         | JSON-RPC 2.0 (stdio / SSE)
                 v                                                       v
+---------------------------------------------------------------------------------------------+
|                                  EDULENS MCP SERVER LAYER                                   |
|                                                                                             |
|   +-------------------------------------------------------------------------------------+   |
|   |                   Authentication & Context Layer (TokenManager)                     |   |
|   |      - Validates Bearer tokens and seeds (dev-token-student-001, etc.)              |   |
|   |      - ContextVar-driven user session isolation (UserContext: student_id)           |   |
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
                          v Direct Invocation                  v Thread-Safe Operations
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

## 3. Supported Transports

EduLens supports three standard transport protocols compliant with the MCP specification:

| Transport | CLI Option | Target Environment | Communication Channel |
| :--- | :--- | :--- | :--- |
| **Server-Sent Events (SSE)** | `--transport sse` | Remote deployment / ChatGPT | HTTP POST for JSON-RPC messages; continuous HTTP GET `/sse` stream for server push. |
| **Streamable HTTP** | `--transport streamable`| Cloud hosting / Modern HTTP | Bidirectional streaming HTTP endpoints via ASGI. |
| **Standard I/O (Stdio)**| `--transport stdio` | Local Claude Desktop / CLI tools| Standard input (`stdin`) and standard output (`stdout`) communicating via JSON-RPC lines. |

---

## 4. MCP Tools Reference

### Tool 1: `ingest_learning_session`
Ingests a complete student-AI conversation, saves the raw conversation transcript to persistent storage, runs the DSP/NLP/ML inference pipeline, and returns real-time learning analytics.

- **Parameters:**
  - `session_id` (`string`, required): Unique identifier for the learning session.
  - `conversation` (`array[object]`, required): List of turns. Each turn must have `role` (`"student"` or `"assistant"`) and `content` (`string`).
  - `topic` (`string`, optional): Curriculum topic discussed (e.g., `"photosynthesis"`, `"calculus"`).
  - `duration_minutes` (`number`, optional): Total session duration.
  - `metadata` (`object`, optional): Supplementary context (e.g., subject, client version).

- **Input JSON Schema:**
```json
{
  "type": "object",
  "properties": {
    "session_id": {"type": "string"},
    "conversation": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": {"type": "string", "enum": ["student", "assistant"]},
          "content": {"type": "string"}
        },
        "required": ["role", "content"]
      }
    },
    "topic": {"type": "string"},
    "duration_minutes": {"type": "number"},
    "metadata": {"type": "object"}
  },
  "required": ["session_id", "conversation"]
}
```

- **Output JSON Schema & Example:**
```json
{
  "status": "success",
  "session_id": "sess_101",
  "student_id": "student_001",
  "topic": "photosynthesis",
  "understanding": {
    "predicted_class": "deep_understanding",
    "class_probabilities": {
      "deep_understanding": 0.88,
      "partial_understanding": 0.10,
      "superficial_understanding": 0.02
    },
    "confidence_score": 0.88
  },
  "predicted_quiz_score": 88.5,
  "expressed_confidence": {
    "confidence_score": 0.85,
    "confidence_level": "high"
  },
  "overall_topic_performance": {
    "topic_name": "photosynthesis",
    "composite_score": 86.2,
    "mastery_tier": "proficient"
  },
  "feature_explanation": {
    "top_contributing_features": [
      {"feature": "vocab_diversity", "direction": "positive"},
      {"feature": "student_word_count", "direction": "positive"}
    ]
  }
}
```

---

### Tool 2: `get_session_analytics`
Retrieves stored analysis results for a previously ingested learning session.

- **Parameters:**
  - `session_id` (`string`, required): Identifier of the session.

- **Output Example:**
```json
{
  "session_id": "sess_101",
  "student_id": "student_001",
  "topic": "photosynthesis",
  "predicted_understanding": "deep_understanding",
  "understanding_probabilities": {"deep_understanding": 0.88},
  "predicted_quiz_score": 88.5,
  "confidence_level": "high",
  "mastery_tier": "proficient",
  "analyzed_at": "2026-11-15T22:30:00Z"
}
```

---

### Tool 3: `get_student_performance`
Retrieves aggregated longitudinal learning metrics across all sessions and quizzes for the student.

- **Parameters:**
  - `student_id` (`string`, optional): In multi-user mode, validated against the authenticated context.
  - `topic` (`string`, optional): Filter metrics by specific curriculum topic.

- **Output Example:**
```json
{
  "student_id": "student_001",
  "total_sessions": 12,
  "total_quizzes": 4,
  "average_understanding": "deep_understanding",
  "average_quiz_score": 84.5,
  "understanding_distribution": {
    "deep_understanding": 8,
    "partial_understanding": 3,
    "superficial_understanding": 1
  },
  "topics_covered": ["photosynthesis", "cellular_respiration", "genetics"],
  "mastery_summary": {
    "photosynthesis": {"composite_score": 86.2, "tier": "proficient"},
    "genetics": {"composite_score": 78.0, "tier": "competent"}
  }
}
```

---

### Tool 4: `get_topic_analytics`
Retrieves curriculum-level analytics for a specific topic across the student's history.

- **Parameters:**
  - `topic` (`string`, required): Name of topic.
  - `student_id` (`string`, optional): Target student.

- **Output Example:**
```json
{
  "topic": "photosynthesis",
  "student_id": "student_001",
  "sessions_count": 3,
  "latest_understanding": "deep_understanding",
  "average_predicted_quiz": 87.2,
  "actual_quiz_average": 85.0,
  "composite_mastery_score": 86.2,
  "mastery_tier": "proficient"
}
```

---

### Tool 5: `list_learning_sessions`
Lists historical learning sessions belonging to the authenticated student with pagination.

- **Parameters:**
  - `student_id` (`string`, optional): Validated against user context.
  - `topic` (`string`, optional): Topic filter.
  - `limit` (`integer`, optional, default: 20): Maximum items to return.
  - `offset` (`integer`, optional, default: 0): Pagination offset.

- **Output Example:**
```json
{
  "student_id": "student_001",
  "total_count": 2,
  "sessions": [
    {
      "session_id": "sess_102",
      "topic": "genetics",
      "created_at": "2026-11-15T21:00:00Z",
      "turns_count": 6,
      "predicted_understanding": "partial_understanding"
    },
    {
      "session_id": "sess_101",
      "topic": "photosynthesis",
      "created_at": "2026-11-15T19:30:00Z",
      "turns_count": 8,
      "predicted_understanding": "deep_understanding"
    }
  ]
}
```

---

### Tool 6: `submit_quiz_results`
Ingests and evaluates a student's post-session quiz responses, updating topic mastery and comparing predicted scores to actual results.

- **Parameters:**
  - `session_id` (`string`, optional): Associated learning session.
  - `topic` (`string`, required): Topic assessed.
  - `answers` (`array[object]`, required): List of questions and chosen answers.
  - `quiz_id` (`string`, optional): Optional quiz identifier.

- **Output Example:**
```json
{
  "quiz_id": "quiz_001",
  "session_id": "sess_101",
  "student_id": "student_001",
  "topic": "photosynthesis",
  "score_percentage": 90.0,
  "total_questions": 5,
  "correct_questions": 4,
  "topic_breakdown": {
    "light_reactions": {"score": 100.0},
    "calvin_cycle": {"score": 80.0}
  },
  "recorded_at": "2026-11-15T22:45:00Z"
}
```

---

## 5. Resources & Prompts

### Resources
- `edulens://curriculum/topics`: Returns standard curriculum topics recognized by EduLens.
- `edulens://health`: Returns server status, active transport, and database connection status.

### Prompts
- `save_learning_session`: Pre-packaged prompt instruction for external AI clients instructing the agent how and when to summarize conversations and invoke `ingest_learning_session`.

---

## 6. Storage Architecture & Persistence Strategy

EduLens implements a modular storage abstraction (`StorageService`) separating raw audit-proof conversation transcripts from derived analytical indicators.

### A. Local Development: SQLite in WAL Mode
By default, the server writes to `data/storage/edulens.db` using SQLite in **Write-Ahead Logging (WAL)** mode with immediate foreign key enforcement.

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
CREATE INDEX IF NOT EXISTS idx_raw_sessions_topic ON raw_sessions(user_id, topic);

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

### Storage Reliability Guarantees
1. **Raw Transcript Persistence First:** In `IngestionService`, the raw conversation is persisted to `raw_sessions` **before** the ML pipeline executes. If machine learning inference fails or throws an exception, the raw session is marked as `processing_failed` with the error trace, but the student's conversation is never lost.
2. **Zero-Duplication Retries:** When a failed session is retried via `retry_failed_session()` or re-ingested with identical content, EduLens reuses the existing `session_id`, updating the existing record without creating duplicate raw rows.
3. **Deterministic Idempotency:** A SHA-256 hash of student messages scoped by `user_id` detects duplicate submissions instantly, returning cached analytics in sub-millisecond time.

### B. Production Persistence: PostgreSQL Architecture
For cloud deployments requiring horizontal scaling across multiple container instances, EduLens's `StorageService` is designed to map directly to PostgreSQL:
- `raw_json`, `full_analytics_json`, and `question_breakdown_json` map directly to PostgreSQL `JSONB` columns with GIN indexing for fast nested queries.
- Thread-safe connection pooling is handled via `asyncpg` / `psycopg3` and PgBouncer.
- Row-Level Security (RLS) can be enabled on `user_id` to guarantee database-level tenant isolation.

---

## 7. Authentication, Multi-User Isolation & Security

EduLens enforces strict user boundaries across all operations.

### Development Mode vs. Production Configuration

| Security Feature | Development / Test Mode (`EDULENS_ENVIRONMENT=development`) | Production Mode (`EDULENS_ENVIRONMENT=production`) |
| :--- | :--- | :--- |
| **Default Dev Tokens** | Seeded (`edulens-dev-token-student-001`, `student_002`) | **NEVER seeded.** Missing token rejects request. |
| **Token Sources** | `Authorization` header, Stdio env, or `?token=` if explicitly allowed | **ONLY `Authorization: Bearer <token>` header.** |
| **Query Parameter Auth** | Allowed only if `EDULENS_ALLOW_QUERY_TOKEN=true` | **STRICTLY REJECTED with HTTP 401.** |
| **Unauthenticated Fallback** | Allowed only if `--no-auth` CLI flag is set | **Forbidden.** Always requires valid credentials. |
| **Multi-User Provisioning** | In-memory token manager | Configured via `EDULENS_API_TOKENS="tok1:uid1,tok2:uid2"` |

### Multi-User Isolation Architecture
1. **Header-Based Authentication:** Remote requests must include `Authorization: Bearer <token>`.
2. **Task-Local User Context (`contextvars`):** Python's `contextvars.ContextVar` binds the authenticated `UserContext` exclusively to the current async task.
3. **Automatic Context Reset:** `AuthMiddleware` sets the context token and enforces `reset_current_user()` inside a `finally` block, ensuring no context bleeding between recycled threads or async tasks.
4. **Forged `student_id` Protection:** If a client passes a `student_id` parameter that differs from the authenticated user identity, the server raises `AuthorizationError` immediately.
5. **Data Access Isolation:** All database reads and writes filter by `user_id = :auth_user_id`. Student A cannot view, list, or modify Student B's data.

---

## 8. Client Setup Guides

### A. ChatGPT Integration: Native MCP vs. GPT Actions

> [!IMPORTANT]
> **Understanding Native MCP vs. GPT Actions:**
> - **Native MCP (Desktop Developer Mode):** Connects to EduLens's MCP server over Server-Sent Events (SSE). Requires ChatGPT Desktop App (macOS or Windows) with Developer Mode enabled (available to Plus, Team, Enterprise, and Edu plans).
> - **GPT Actions (Web Custom GPTs):** Standard ChatGPT Web Custom GPTs do **not** speak the JSON-RPC 2.0 MCP protocol. They communicate via REST OpenAPI 3.0. For web users, EduLens exposes standard REST endpoints at `/api/pipeline` and `/api/predict` described below.

#### Option 1: Native MCP Connectivity (ChatGPT Desktop Developer Mode)
1. Launch **ChatGPT Desktop App** on macOS or Windows.
2. Go to **Settings** -> **Developer** -> **MCP Servers** -> **Add Server**.
3. Fill in the connection parameters:
   - **Name:** `EduLens Learning Analytics`
   - **Transport:** `Server-Sent Events (SSE)`
   - **URL:** `http://127.0.0.1:8001/sse` (or your deployed `https://<your-domain>/sse`)
   - **Headers:**
     - Key: `Authorization`
     - Value: `Bearer <your-student-api-token>`
4. Click **Save**. ChatGPT initiates the MCP protocol handshake, queries `tools/list`, and discovers all 6 EduLens tools.
5. In your conversation prompt:
   ```text
   You are an educational AI tutor. When we finish discussing a topic,
   call the ingest_learning_session tool to save the conversation to EduLens
   and review my learning analytics with me.
   ```

#### Option 2: GPT Actions Bridge (ChatGPT Web Custom GPTs)
For standard web accounts without MCP developer mode:
1. Open **Explore GPTs** -> **Create a GPT** -> **Configure** -> **Actions** -> **Create new action**.
2. Set Authentication to `API Key` (Type: `Bearer`) and provide your API token.
3. Import the OpenAPI specification from `http://<your-server>:8000/openapi.json`.
4. ChatGPT will invoke EduLens's `/api/pipeline` REST endpoint.

---

### B. Claude Desktop Configuration

Claude Desktop supports native MCP out-of-the-box via both local Stdio and remote SSE transports.

#### 1. Local Stdio Transport (Recommended for Desktop)
Edit your Claude Desktop configuration:
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

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

#### 2. Remote SSE Transport
For connecting Claude Desktop to a hosted remote EduLens server:
```json
{
  "mcpServers": {
    "edulens-remote": {
      "url": "http://127.0.0.1:8001/sse",
      "headers": {
        "Authorization": "Bearer edulens-dev-token-student-001"
      }
    }
  }
}
```

---

## 9. Local Development & Deployment

### Running the Server Locally

```bash
# Activate virtual environment
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# 1. Start with SSE Transport (Default for remote clients / ChatGPT)
python -m backend.mcp_server.server --transport sse --port 8001

# 2. Start with Streamable HTTP Transport
python -m backend.mcp_server.server --transport streamable --port 8001

# 3. Start with Stdio Transport (For Claude Desktop integration)
python -m backend.mcp_server.server --transport stdio
```

### Production Deployment (Docker / VPS / Cloud)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV EDULENS_ENVIRONMENT=production
ENV EDULENS_MCP_TRANSPORT=sse
ENV EDULENS_MCP_PORT=8001
ENV EDULENS_AUTH_ENABLED=true
ENV EDULENS_STORAGE_DIR=/app/data/storage
EXPOSE 8001
CMD ["python", "-m", "backend.mcp_server.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8001"]
```

---

## 10. Automated Testing & Verification

EduLens includes a full test suite verifying all MCP endpoints, transport layers, security isolation, and live client sessions.

```bash
# Run all tests (75 passing tests)
.\.venv\Scripts\pytest tests/ -v

# Run MCP server unit & integration tests (33 tests)
.\.venv\Scripts\pytest tests/test_mcp_server.py -v

# Run live MCP client session E2E test
.\.venv\Scripts\pytest tests/test_mcp_e2e.py -v
```

**Verification Results:**
- `tests/test_mcp_server.py`: **33/33 tests passing** (authentication, tool discovery, multi-user isolation, concurrency, retry resilience, quiz updates).
- `tests/test_mcp_e2e.py`: **1/1 test passing** (live subprocess client session over Stdio).
- **Overall Project Suite:** **75/75 tests passing with 0 errors**.


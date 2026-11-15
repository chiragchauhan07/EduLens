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

## 6. Storage Architecture & Schema

All persistent data is stored in `data/storage/edulens.db` using SQLite in **Write-Ahead Logging (WAL)** mode for concurrent, thread-safe access.

```sql
-- Raw Conversational Records (Audit-Proof)
CREATE TABLE IF NOT EXISTS raw_sessions (
    session_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    topic TEXT,
    duration_minutes REAL,
    raw_conversation_json TEXT NOT NULL,
    metadata_json TEXT,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_raw_student ON raw_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_raw_hash ON raw_sessions(content_hash);

-- Derived Machine Learning Analytics
CREATE TABLE IF NOT EXISTS analytics_results (
    session_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    topic TEXT,
    predicted_understanding TEXT NOT NULL,
    understanding_probabilities_json TEXT,
    confidence_score REAL,
    predicted_quiz_score REAL,
    expressed_confidence_json TEXT,
    overall_performance_json TEXT,
    feature_explanation_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES raw_sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analytics_student ON analytics_results(student_id);
CREATE INDEX IF NOT EXISTS idx_analytics_topic ON analytics_results(topic);

-- Quiz Submissions & Performance
CREATE TABLE IF NOT EXISTS quiz_submissions (
    submission_id TEXT PRIMARY KEY,
    session_id TEXT,
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    score_percentage REAL NOT NULL,
    total_questions INTEGER NOT NULL,
    correct_questions INTEGER NOT NULL,
    question_details_json TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_quiz_student ON quiz_submissions(student_id);
```

---

## 7. Authentication & Multi-User Isolation

Authentication is enforced across both remote HTTP and Stdio transports:

1. **Bearer Token Validation:** Remote HTTP/SSE requests require standard `Authorization: Bearer <token>` headers.
2. **TokenManager Resolution:** The token is resolved to a `UserContext(student_id=...)`.
3. **Async Context Isolation:** Python's `contextvars.ContextVar` binds the authenticated `student_id` to the request coroutine.
4. **Data Isolation:** All storage queries filter by `student_id = :authenticated_student_id`. A student cannot access, list, or overwrite another student's sessions.
5. **Pre-Seeded Development Keys:**
   - `edulens-dev-token-student-001` -> mapped to `student_001`
   - `edulens-dev-token-student-002` -> mapped to `student_002`

---

## 8. Client Setup Guides

### A. ChatGPT Configuration (Custom GPT / Action)

1. Navigate to **ChatGPT** -> **Explore GPTs** -> **Create a GPT**.
2. Under the **Configure** tab:
   - **Name:** EduLens Learning Companion
   - **Instructions:**
     ```
     You are an educational AI tutor. Help students explore topics deeply.
     When a student completes a discussion on a concept or asks to conclude the session,
     call the ingest_learning_session tool to save the conversation and review their
     understanding metrics with them.
     ```
3. Under **Actions**, click **Create new action**:
   - **Authentication:** `API Key` -> Auth Type: `Bearer`.
   - **API Key:** Paste your student token (e.g. `edulens-dev-token-student-001`).
   - **Server URL:** `http://your-server-ip:8001/` (or your deployed domain).
   - **OpenAPI Schema / Action URL:** Import the MCP OpenAPI specification from your running EduLens server endpoint (`/openapi.json`).

### B. Claude Desktop Configuration

Claude Desktop connects seamlessly via either local Stdio or remote SSE.

#### 1. Local Stdio Transport (Recommended for Desktop)
Edit your Claude Desktop configuration file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the EduLens MCP server entry:
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
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1

# 1. Start with SSE Transport (Default on port 8001)
python -m backend.mcp_server.server --transport sse --port 8001

# 2. Start with Streamable HTTP Transport
python -m backend.mcp_server.server --transport streamable --port 8001

# 3. Start with Stdio Transport (For Claude Desktop integration)
python -m backend.mcp_server.server --transport stdio
```

### Production Deployment (Railway / Docker / VPS)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV EDULENS_MCP_TRANSPORT=sse
ENV EDULENS_MCP_PORT=8001
ENV EDULENS_STORAGE_DIR=/app/data/storage
EXPOSE 8001
CMD ["python", "-m", "backend.mcp_server.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8001"]
```

---

## 10. Automated Testing & Verification

EduLens includes a full test suite verifying all MCP endpoints, transport layers, and end-to-end client sessions.

```bash
# Run all tests
pytest tests/ -v

# Run MCP server unit & integration tests
pytest tests/test_mcp_server.py -v

# Run live MCP client session E2E test
pytest tests/test_mcp_e2e.py -v
```

**Verification Results:**
- `tests/test_mcp_server.py`: 24/24 tests passing.
- `tests/test_mcp_e2e.py`: Live subprocess ClientSession passing.
- **Overall Project Suite:** 66/66 tests passing with 0 errors.


# EduLens

**NLP and Machine Learning-Based Student Learning Intelligence and Performance Analytics Platform**

## Overview

EduLens is not another AI tutor. Students already learn topics using external LLM clients like ChatGPT or Claude — EduLens is the **intelligence and analytics layer** built around that existing workflow.

After a learning session, the student sends the conversation to EduLens through an MCP (Model Context Protocol) integration. EduLens then automatically:

- Analyzes the conversation using NLP to extract topics, concepts, confusion, confidence, sentiment, and cognitive level
- Generates a targeted quiz (10–15 questions) using the Gemini API, based on the student's actual weak areas
- Collects quiz performance and combines it with conversational signals
- Applies Machine Learning to predict topic mastery, performance, and learning risk
- Maintains a longitudinal model of each student's learning state over time
- Presents everything through an analytics dashboard with personalized recommendations

## Core Idea

```
Student learns via ChatGPT/Claude
        ↓
"Send this session to EduLens" (MCP)
        ↓
Automated Data Pipeline (NLP + Feature Engineering)
        ↓
Targeted Quiz Generation (Gemini API)
        ↓
Quiz Attempt + Combined Data
        ↓
ML Models (Mastery, Performance, Risk Prediction)
        ↓
Longitudinal Student Model → Dashboard & Reports
```

## Why This Project

Most student-facing AI tools focus on being the tutor. EduLens instead asks: **once a student has already learned something from an LLM, how do we measure, assess, and track whether real learning happened — over time, not just in one session?**

The project is built around a clear architectural principle:

> Deterministic code for data processing + ML models for prediction + LLMs only where generative/reasoning capability is actually needed.

This keeps the system explainable and production-viable, rather than routing everything through an LLM agent.

## Key Features (Planned)

- 🔌 **MCP Server** — receives learning-session context from external LLM clients
- 🧠 **NLP Analysis** — topic/concept extraction, confusion & confidence detection, cognitive-level classification
- 📝 **Targeted Quiz Generation** — Gemini-powered, based on extracted weak concepts rather than generic questions
- 📊 **ML Layer** — mastery prediction, performance prediction, learning-risk identification
- 📈 **Longitudinal Student Model** — tracks mastery, confidence, and confusion across sessions over time
- 📉 **Analytics Dashboard** — before/after learning comparisons, topic-wise performance, personalized recommendations

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | React / Next.js, TypeScript, Tailwind CSS, Recharts |
| Backend | Python, FastAPI, Pydantic, SQLAlchemy |
| MCP | Python MCP SDK |
| NLP | spaCy, Hugging Face Transformers, sentence-transformers |
| Data Science / ML | Pandas, NumPy, scikit-learn, XGBoost |
| Generative AI | Gemini API |
| Database | PostgreSQL, Redis (optional) |
| Deployment | Docker, Docker Compose |

## Project Status

🚧 In active development — see project board / issues for current phase.

Development follows an incremental, phased approach (backend → MCP integration → data pipeline → NLP → quiz generation → ML → analytics → Dockerization → hardening) rather than building everything at once.

## Data Privacy

Student conversation data is treated as sensitive. EduLens collects only what is necessary for analysis, secures stored data and API credentials, and supports data deletion. The system does not assume unrestricted access to a user's conversation history — data is only processed when explicitly shared through the supported MCP integration.

## License

TBD

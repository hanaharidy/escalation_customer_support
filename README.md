# AI Customer Support System

An intelligent multi-agent customer support system built for e-commerce. Instead of a simple chatbot that either answers or gives up, this system thinks through problems in layers — trying everything it can before ever escalating to a human agent.

---

## The Idea

Most chatbots do one of two things: they either answer from a script, or they immediately say "let me connect you with an agent." This project tries to do what a good support agent would actually do — understand the problem, look it up, take action if possible, ask clarifying questions if needed, suggest alternatives, and only then escalate if nothing worked.

The system is built around six specialized agents that each handle one part of that process, coordinated by LangGraph into a coherent workflow.

---

## How It Works

When a customer sends a message, it goes through this pipeline:

```
Customer message
      │
      ▼
Conversation Manager   — understands what the customer wants
      │
      ▼
Knowledge Base Agent   — searches documentation for an answer (RAG)
      │
      ▼
Action Agent           — looks up orders, processes refunds, checks returns
      │
      ▼
Sentiment Analyzer     — detects frustration level
      │
      ▼
Escalation Decider     — decides if human help is actually needed
      │
      ▼
Learning Agent         — logs the resolution for future improvement
```

The system tries to resolve issues at four levels before escalating:
- **Level 1** — Answer from the knowledge base
- **Level 2** — Execute an action (order lookup, refund, return check)
- **Level 3** — Ask the customer for more details
- **Level 4** — Suggest alternative contact channels
- **Level 5** — Connect to a human agent (last resort)

---

## The Six Agents

**Conversation Manager** reads the customer's message and figures out what they want — tracking an order, requesting a refund, asking a policy question, or something else. It also extracts useful details like order IDs. If a customer asks for a human agent, it tries to help them first before escalating.

**Knowledge Base Agent** runs a semantic search over the documentation using RAG (Retrieval-Augmented Generation). It finds the most relevant chunks from the knowledge base, passes them to the LLM as context, and generates a grounded answer. Every response includes a confidence score and the source documents used.

**Action Agent** handles transactional requests by calling the order database directly. It can look up real order status, initiate refunds, and check return eligibility — all connected to a SQLite database seeded with realistic demo data.

**Sentiment Analyzer** reads the emotional tone of each message. It tracks how many consecutive turns the customer has been frustrated or angry, which feeds into the escalation decision.

**Escalation Decider** looks at everything — sentiment history, confidence scores, resolution level, whether alternatives were already suggested — and makes the final call on whether to escalate. Human escalation only happens when the bot has genuinely exhausted its options.

**Learning Agent** closes the feedback loop. When a human agent resolves an escalated ticket through the dashboard, that resolution gets embedded and stored in ChromaDB. Future customers asking similar questions will get the benefit of that knowledge.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + WebSockets |
| Agent Orchestration | LangGraph |
| Vector Database | ChromaDB |
| Order Database | SQLite |
| Primary LLM | Groq (llama-3.3-70b-versatile) |
| Fallback LLM | Google Gemini (gemini-1.5-flash) |
| Embeddings | all-MiniLM-L6-v2 (runs locally) |

The embedding model runs entirely on your machine — no API calls needed for search.

---

## Project Structure

```
ai_support_system/
├── backend/
│   ├── main.py                    # FastAPI app and WebSocket endpoint
│   ├── config.py                  # All settings in one place
│   ├── agents/
│   │   ├── conversation_manager.py
│   │   ├── knowledge_base_agent.py
│   │   ├── sentiment_analyzer.py
│   │   ├── action_agent.py
│   │   ├── escalation_decider.py
│   │   └── learning_agent.py
│   ├── graph/
│   │   ├── state.py               # Shared state flowing between agents
│   │   └── workflow.py            # LangGraph pipeline assembly
│   ├── rag/
│   │   ├── ingestion.py           # Load docs, chunk, embed, store
│   │   └── retriever.py           # Semantic search interface
│   ├── actions/
│   │   ├── database.py            # SQLite setup, loaded from seed_data.json
│   │   └── order_actions.py       # Order lookup, refund, return check
│   ├── memory/
│   │   └── session_store.py       # Per-session conversation history
│   ├── escalation/
│   │   └── ticket_manager.py      # Escalation ticket creation and storage
│   ├── llm/
│   │   └── client.py              # LLM abstraction (Groq / Gemini / Ollama)
│   └── models/
│       └── schemas.py             # API request/response models
├── frontend/
│   └── app.py                     # Streamlit UI (no AI logic here)
├── data/
│   ├── knowledge_base/            # FAQ, return policy, shipping info
│   │   ├── faqs.md
│   │   ├── return_policy.md
│   │   └── shipping_info.md
│   └── seed_data.json             # Demo orders and customers
├── .env.example                   # Environment variable template
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/ai-support-system.git
cd ai-support-system
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Copy `.env.example` to `.env` and fill in your API keys:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).
Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

**5. Start the backend**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

On first run, the system will:
- Initialize the SQLite database with demo orders
- Load and embed the knowledge base into ChromaDB
- Download the embedding model (~90MB, cached after first run)

**6. Start the frontend** (in a new terminal)
```bash
source venv/bin/activate
python -m streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501).

---

## Try It Out

Once running, here are some things worth trying:

- `Where is my order ORD-123?` — looks up a real order from the database
- `I want a refund for ORD-456` — initiates a refund and updates order status
- `What is your return policy?` — answered from the knowledge base with source citations
- `My package arrived damaged` — triggers return eligibility check
- `I want to speak to a human` — bot tries to help first, escalates if you insist

To see the full escalation flow, send a vague message like "I have a problem" and keep responding without giving specific details. Watch the resolution level climb through clarification → alternatives → human escalation.

---

## The Learning Loop

When a conversation gets escalated, a ticket is created with the full conversation history. Switch to the **Agent Dashboard** tab to see open tickets. After a human agent writes a resolution and clicks "Resolve", that answer gets embedded and stored in ChromaDB. The next customer asking a similar question will get the benefit of that resolution — the system genuinely gets smarter over time.

---

## Demo Orders

These orders are pre-loaded in the database:

| Order ID | Status | Items |
|----------|--------|-------|
| ORD-123 | Shipped | Wireless Headphones |
| ORD-456 | Processing | Laptop Stand, USB-C Hub |
| ORD-789 | Delivered | Phone Case |
| ORD-999 | Cancelled | Screen Protector |
| ORD-111 | Shipped | Mechanical Keyboard |
| ORD-222 | Delivered | Wireless Mouse |

---

## Known Limitations

Session history and escalation tickets live in memory — they reset if the server restarts. ChromaDB and the SQLite database persist on disk and survive restarts. In a production setup, session state would move to Redis and tickets would go into a proper database.

The mock contact channels (email, phone number, support URL) in the alternatives suggestions are placeholders and don't connect to real services.

---

## Architecture Notes

**Separation of concerns** is strictly maintained — Streamlit handles display only, all AI logic lives in FastAPI. This means you could swap the frontend entirely without touching the backend.

**The LLM is swappable** — changing `LLM_PROVIDER` in `.env` switches between Groq, Gemini, and local Ollama. No code changes needed.

**Embeddings are local** — `all-MiniLM-L6-v2` runs on your machine. The embedding step never makes an API call, which means RAG retrieval is fast and free regardless of how many queries you run.
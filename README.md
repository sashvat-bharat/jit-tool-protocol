# JIT (Just-in-Time) Tool Spawning Protocol

> A server-side routing architecture that lets AI agents scale to thousands of tools without context bloat, hallucinations, or spiraling API costs.

**Author:** Akshat Dwivedi - Founder & CEO, [Sashvat Bharat](https://sashvat.com)<br>
**Date:** March 2026

---

## The Problem

Today's multi-agent AI systems stuff *every* tool definition into the LLM's system prompt — whether the user needs 1 tool or 150. This leads to:

- 💸 **Massive token costs** on every single query (even "Hello")
- 🐌 **High latency** from processing bloated prompts
- 🤯 **Hallucinations** from "Lost in the Middle" attention degradation
  
## The Solution

The JIT Protocol decouples **tool discovery** from **tool execution** using a three-tier pipeline:



```
User Query
    │
    ▼
┌──────────────────────────┐
│  1. Vector DB (Coarse)   │  Embed query → Cosine similarity search
│     Returns Top K tools  │  against pre-embedded tool descriptions
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  2. SLM Router (Fine)    │  A fast Small Language Model filters
│     Returns 1–3 tools    │  Top K down to exact tools needed
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  3. Heavy LLM (Execute)  │  Only the selected tool schemas are
│     Executes the task    │  injected → near-zero context bloat
└──────────────────────────┘
```


**Result:** Whether your agent has 10 tools or 10,000 tools, the cost per query stays **flat**. Our experiments show **~88.8% token savings** compared to the traditional approach with just 50 tools — a saving that only grows as the registry scales.


---

## Quick Start :

### Prerequisites -

- A [Google Gemini API Key](https://aistudio.google.com/apikey)
  
### Installation -

```bash
# Clone the repo
git clone https://github.com/sashvat-bharat/jit-tool-protocol.git
cd jit-tool-protocol

# Install dependencies
pip install google-genai numpy python-dotenv

# Set up your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run the Demo -

```bash
python jit_router.py
```
  
This executes the full JIT pipeline against a built-in registry of 10 tools with the test query:  

*"What is the current trading price of Apple, and email it to my CFO."*
  

**Expected output:**

```
-> Embedding tools into memory...

[USER QUERY]: What is the current trading price of Apple, and email it to my CFO.
-> Vector DB Found Top 5: ['get_stock_price', 'send_email', ...]
-> SLM (Gemini 3 Flash) evaluating true intent...
-> SLM Selected: ['get_stock_price', 'send_email']
  
[SUCCESS] Final Payload ready for the Heavy LLM:
[
  { "type": "function", "name": "get_stock_price", ... },
  { "type": "function", "name": "send_email", ... }
]
```


---
## Repository Structure

```
jit-tool-protocol/
├── jit_router.py            # Reference implementation of the JIT Protocol
├── fifty_tools.json         # Registry of 50 tool definitions for benchmarking
├── jit_research_paper.md     # Full research paper (Markdown)
├── jit_research_paper.pdf    # Full research paper (PDF)
├── .env.example              # Environment variable template
├── .gitignore
└── images/                   # Diagrams and screenshots
```

| File                    | Purpose                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `jit_router.py`         | End-to-end implementation: tool embedding, vector search, SLM routing, and payload assembly |
| `fifty_tools.json`      | A 50-tool registry used for the token-savings benchmark in the paper                        |
| `jit_research_paper.md` | The full paper with architecture details, math, and experimental results                    |

---
## How It Works (In Brief)

1. **Vector Indexing** — Each tool's natural-language description is embedded (via `gemini-embedding-001`) and stored in a vector index at startup time.
  
2. **Coarse Retrieval** — When a user query arrives, it's embedded and compared via **cosine similarity** against all tool embeddings. The **Top K** (default 5) closest tools are retrieved.

3. **SLM Semantic Routing** — The Top K short descriptions are sent to a fast SLM (`gemini-3-flash-preview`) with a strict routing prompt. The SLM returns only the **exact tool names** needed (typically 1–3).

4. **Payload Injection** — The full JSON schemas of only the selected tools are fetched from the registry and injected into the Heavy LLM's prompt. The Heavy LLM executes the task with near-zero context bloat.

> 📖 **For the full deep-dive** — architecture diagrams, mathematical cost analysis, multi-turn conversation handling, and experimental benchmarks - read the research paper :
> 
> **PDF :** [jit_research_paper.pdf](https://github.com/sashvat-bharat/jit-tool-protocol/blob/main/jit_research_paper.pdf)
> **Markdown :** [jit_research_paper.md](https://github.com/sashvat-bharat/jit-tool-protocol/blob/main/jit_research_paper.md)

---
## Experimental Results

| Metric                       | Traditional (50 Tools) | JIT Protocol         |
| ---------------------------- | ---------------------- | -------------------- |
| Tool definitions sent to LLM | All 50 schemas         | 1–3 selected schemas |
| Tool-related tokens consumed | ~5,000 tokens          | ~560 tokens          |
| **Token savings**            | —                      | **~88.8% reduction** |

As the tool registry scales to 200, 500, or 10,000+ tools, the traditional approach's cost grows **linearly** while the JIT Protocol's cost stays **flat**.

---
## Tech Stack -

| Component       | Technology                                                            |
| --------------- | --------------------------------------------------------------------- |
| Embedding Model | `gemini-embedding-001`                                                |
| SLM Router      | `gemini-3-flash-preview`                                              |
| Vector Search   | In-memory cosine similarity (swappable with pgvector, Pinecone, etc.) |
| Language        | Python 3.9+                                                           |

---
## License

This project is open-sourced for research and educational purposes.

---

## 📝 Citation

  If you use this protocol in your work, please cite:

```bibtex
@article{jit_protocol_2026,
  title = {The JIT (Just-in-Time) Tool Spawning Protocol: A Server-Side Routing Architecture for Infinite-Tool AI Agents},
  author = {Akshat Dwivedi},
  year = {2026},
  month = {March},
  publisher = {Sashvat Bharat},
  url = {https://github.com/sashvat-bharat/jit-tool-protocol}
}
```

---
## ⭐ Star History

<a href="https://star-history.com/#sashvat-bharat/jit-tool-protocol&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sashvat-bharat/jit-tool-protocol&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sashvat-bharat/jit-tool-protocol&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sashvat-bharat/jit-tool-protocol&type=Date" width="100%" />
 </picture>
</a>

---

> **Thanks for stopping by!** If this project helped you, consider giving it a ⭐ — it helps others discover the work.  

> Built by [Akshat Dwivedi](https://github.com/Akshat-Dwivedi-52) at [Sashvat Bharat](https://sashvat.com).
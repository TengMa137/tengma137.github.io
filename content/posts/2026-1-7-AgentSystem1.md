---
title: 'Choosing the right Frameworks and tools for Agents'
date: 2026-01-07
tags:
  - Agent system
  - AI in prodcution
  - Framework and toolchain selection
---


In the last post, I focused on why most AI agent demos struggle in production. This one addresses the next question teams ask once they accept those constraints:

**What frameworks, retrieval methods, and abstractions should we actually use?**

I questioned myself a lot when building my first agent system. The honest answer is still inconvenient:

**It depends on the system you’re trying to run, not the tools you like, or the blog posts you’ve read.**

It’s a way to reason about *tradeoffs* - and to recognize when an abstraction is helping versus when it’s just hiding complexity. 

## Agents Are Just Systems of LLM Calls

The first useful reframing is to stop treating “agents” as a special category.

Mechanically, an agent system is just coordinated LLM calls with shared context, routing logic, tool invocation, retries, and termination conditions. Once you see that, frameworks stop feeling magical and start feeling like what they are: **opinions about control flow, state, and observability**.


## Writing Your Own Agent Loop vs Using a Framework

A few hundred lines of Python can suffice to implement the core pieces (model calls, a reasoning loop, and tool execution, optionally with memory/state control) and give you deep insight into how agents work under the hood. The core logic of an agent is just a controlled loop over reasoning and actions, and it's perfectly fine for a small amount of users or just learning concepts.

Frameworks, on the other hand, earn their keep once complexity compounds - multiple tools, retries, parallelism, observability, structured outputs, or multi-agent coordination. They trade transparency for speed and consistency.
In practice, the trade-off looks like this:
- Custom agent loop: Maximum control and debuggability, at the cost of more upfront effort and ongoing maintenance. You own the failure modes, but you also understand them deeply.
- Agent framework: Faster time-to-demo and shared infrastructure, but with opinionated abstractions that can obscure control flow and make edge cases harder to reason about.

A custom loop forces you to confront what actually breaks: where tokens are wasted, how retries amplify cost, and how small prompt changes ripple through the system. Even if you later adopt a framework, building a minimal agent once gives you a mental model most teams never quite develop - and a sharper sense of what frameworks truly provide versus what they hide.


## Comparing Popular Agent Framework Styles

Most agent frameworks differ less in *capability* than in *philosophy*. They all wrap the same primitives - model calls, tools, state, and control flow, but make very different tradeoffs about how explicit those pieces should be. At a high level:

* **LangChain-style frameworks** optimize for velocity and ecosystem breadth. They make it easy to wire models, tools, and memory together, but often at the cost of opaque control flow.
* **Graph- or DAG-based systems** make state transitions explicit. You do more upfront design, but you get predictability, debuggability, and replayability.
* **Planner-heavy / auto-agent approaches** maximize autonomy, letting the model decide what to do next with minimal orchestration. These are powerful for demos and research, but brittle in production.

Frameworks are rarely the bottleneck. Misunderstanding what they abstract away usually is.

### Agent Framework Comparison (Real-World, by 2025)

| Framework               | Style / Philosophy             | Strengths                                                  | Tradeoffs                                      | Typical Use                                       |
| ----------------------- | ------------------------------ | ---------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **LangChain**           | Chain & agent abstractions     | Huge ecosystem, fast prototyping, broad model/tool support | Control flow can be implicit and hard to trace | Prototyping, internal tools, early-stage products |
| **LangGraph**           | Explicit state machine / graph | Deterministic execution, resumability, production-friendly | More upfront design, less “plug-and-play”      | Complex workflows, long-running agents            |
| **LlamaIndex**          | Data-centric agent framework   | Strong RAG, indexing, retrieval primitives                 | Agent logic less flexible than LangChain       | Knowledge-heavy apps, enterprise RAG              |
| **PydanticAI**          | Typed, schema-first agents     | Strong structure, validation, predictable outputs          | Less flexible for exploratory agents           | Production systems, tool-heavy backends           |
| **CrewAI**              | Role-based multi-agent         | Simple mental model, good demos, fast setup                | Coordination logic can get implicit            | Personal projects, agent “teams”, experiments     |
| **AutoGen**             | Conversational multi-agent     | Flexible agent-to-agent interaction                        | Hard to reason about, non-deterministic        | Research, simulations, experimentation            |

---

Teams could use more than one framework - a minimal loop or LangGraph-style core for production paths, and a higher-level framework for experimentation and iteration.


Here’s a reorganized version that **keeps the substance**, **reduces table fatigue**, and pushes the reader toward clearer mental models. I’ve collapsed tables into bullets where pattern recognition matters, and kept **one table per section only where it genuinely helps**.


## Retrieval Is Task-Specific, Not Vector-by-Default

One of the most common production mistakes is treating embeddings as the default retrieval solution. They aren’t.

Retrieval is a *systems design problem*, not a modeling one. Different tasks demand different primitives, and forcing everything through a vector index usually makes systems slower, noisier, and harder to debug. Here are some example use cases:

| Search Target / Task Type                    | Best Retrieval Method         | Why                                              |
| -------------------------------------------- | ----------------------------- | ------------------------------------------------ |
| Code search                                  | Lexical (BM25, trigrams)      | Token-level precision and symbol matching matter |
| Exact record lookup                          | SQL / key-value               | Deterministic, cheap, predictable                |
| Structured entities (users, orders, configs) | SQL + indexes                 | Clear schema, no semantic ambiguity              |
| Metadata filtering                           | SQL / column filters          | Faster and more accurate than embeddings         |
| Log / trace search                           | Lexical + time filters        | Ordering and exact matches dominate              |
| FAQ / doc QA                                 | Embeddings                    | Semantic similarity helps recall                 |
| Natural-language to data                     | Hybrid (filters + embeddings) | Structure first, semantics second                |
| Long-form research                           | Hybrid + reranking            | Balance recall and precision                     |
| Mixed or unknown queries                     | Hybrid                        | Safest default in production                     |

Robust systems usually start with **deterministic retrieval**, layer in semantic search only where it clearly helps, and then reconcile results with ranking or filtering. This approach is cheaper, faster, and far more debuggable than “vector-everything” pipelines.


### RAG Pipelines Are a Spectrum, Not a Pattern

Retrieval-augmented generation isn’t a single recipe - it’s a set of tradeoffs between simplicity, precision, and control.

Most production systems evolve along this path:

* **Simple RAG** (embed → retrieve → prompt)
  Fast to build, easy to demo, but noisy and opaque.
* **Filtered RAG** (metadata filters + retrieval)
  Adds precision, requires schema discipline.
* **Hybrid RAG** (lexical + embeddings + reranking)
  More moving parts, but significantly better quality and debuggability.
* **Multi-stage RAG** (iterative retrieval + reasoning)
  High recall, high latency, usually reserved for research or complex workflows.

### Databases Are Part of the Retrieval Design

Here the table actually earns its place, because it maps constraints to reasonable defaults:

| Context                           | Reasonable Choice   |
| --------------------------------- | ------------------- |
| Local / small datasets            | SQLite + FTS5       |
| Local vector search               | SQLite + sqlite-vec |
| Existing Postgres stack           | Postgres + pgvector |
---

There are many great opensource project optimized for search, such as **LanceDB**, a strong choice when you want fast local or cloud vector search with good developer ergonomics. Useful once vector workloads become central - not required on day one.
**Meilisearch** is excellent for search-first applications (docs, catalogs, dashboards). If search is a core product feature rather than a supporting capability, it can replace large parts of a custom retrieval stack.

The goal isn’t picking the “best” database up front. It’s choosing something that lets the retrieval system evolve as usage becomes real, instead of starting at maximum complexity and discovering later that most queries never needed embeddings in the first place.


## Monitoring and Evaluation Are Not Optional

Once agents and RAG systems leave notebooks, **observability becomes part of the architecture**, not an afterthought.

Every popular framework already emits OpenTelemetry traces. That means you have three realistic options:

### Monitoring & Evaluation Approaches

| Observation Backend         | Examples                            | Strengths                      | Tradeoffs                | When It Makes Sense     |
| --------------------------- | ----------------------------------- | ------------------------------ | ------------------------ | ----------------------- |
| **Framework-native**        | LangSmith, LlamaIndex Observability | Zero-friction, framework-aware | Framework lock-in        | Single-framework stacks |
| **Framework-agnostic**      | Langfuse, Helicone, W&B             | Cross-framework visibility     | Integration effort       | Mixed agent stacks      |
| **OTel-compatible backend** | Jaeger, Grafana Tempo, Honeycomb    | Vendor-neutral, infra-native   | Lowest-level abstraction | Mature platforms        |
---

Orthogonally, you can add custom evaluation metric as needed. At minimum, you want visibility into:

* prompt versions
* retrieved documents
* tool calls
* latency and token usage
* failure and retry paths

Without this, you’re tuning blind. With it, many “agent problems” turn out to be simple retrieval or prompt issues.


## MCP Is a Coordination Tool, Not an Intelligence Upgrade

Model Context Protocol (MCP) is best understood as a boundary: a contract between agents and tools.

It shines when multiple agents share tooling, when execution must be clearly separated from reasoning, or when tools evolve independently. It’s overkill for tightly scoped systems where simplicity and latency dominate. 

Like most abstractions, MCP solves coordination problems - not reasoning problems.


## Tool Choice Is About Constraints, Not Fashion

Across production systems, tools change constantly. Constraints don’t.

* Latency budgets.
* Cost ceilings.
* Data freshness.
* Observability.
* Failure tolerance.

Good tools make these constraints explicit and manageable. Bad ones hide them until the system becomes fragile.

The goal isn’t to pick the “best” framework or RAG pipeline. It’s to pick ones that fail in ways you can understand, debug, and recover from.


---
In the next post, I’ll walk through a concrete experiment: building a local agent with retrieval, monitoring, and evaluation that fits *my* constraints - not a generic benchmark.

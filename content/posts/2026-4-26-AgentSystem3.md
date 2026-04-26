---
title: 'Evaluating RAG Pipelines: What Actually Helped'
date: 2026-04-26
tags:
  - AI Agent system
  - Retrieval Augmented Generation
  - RAG Evaluation
  - Context Engineering
---

In the previous posts, I wrote about agent systems from the architecture side: how to choose frameworks, how to structure agent loops, how to think about context, and how my local research agent uses a shared RAG layer.

This post is about the next step: evaluation.

I wanted to answer a simple question:

**Which retrieval pipeline actually works better for paper question answering?**

Not which one feels more elegant. Not which one is more fashionable. Not which one gives a nice demo on a single document. I wanted a comparison across lexical retrieval, vector retrieval, hybrid retrieval, and my hierarchical Zoom retrieval pipeline.

The result was more interesting than I expected. The strongest lesson was not "vector search is bad" or "lexical search is enough." The lesson was that retrieval quality depends heavily on the second stage: how the system decides whether the first evidence set is enough, and how it reranks the candidate evidence before sending it to the answer model.

## The setup

The evaluation uses an Arxiv subset from Open RAGBench. The task is question answering over papers, with gold section-level evidence and reference answers. I compared four pipelines:

- **Lexical**: flat retrieval over predefined section chunks.
- **Vector**: embedding retrieval over the same section chunks.
- **Hybrid**: lexical plus vector retrieval.
- **Zoom**: one structured full-paper document per paper, indexed as a hierarchy of paper -> section -> sub-section nodes.

The dataset itself matters a lot here.

These are not noisy customer support tickets, casual chat logs, or vague product questions. They are academic papers. The input text is relatively high quality, carefully edited, and full of stable terms: method names, dataset names, mathematical symbols, section titles, abbreviations, citations, and domain-specific phrases. Many questions also reuse the same vocabulary as the source paper. In this setting, lexical retrieval is not a naive baseline. It is a very strong default.

This is exactly the kind of corpus where BM25-style search can look surprisingly hard to beat. If the answer depends on a phrase like "conformal prediction", "holonomy group", "symplectic current", or the name of a dataset, matching the term precisely is often more useful than embedding it into a smoother semantic space. Vector search shines when the user describes a concept indirectly or uses different language from the document. Academic QA often does the opposite: it asks about named concepts using the paper's own vocabulary.

That is why I do not read these results as "vector search is bad." I read them as: **for clean academic text with stable terminology, lexical retrieval is a proper go-to method, not just a cheap fallback.**

The important difference is that the flat baselines retrieve over already chunked sections, while Zoom sees each paper as one structured document and navigates the tree.

That means Zoom is solving a harder version of the retrieval problem. It has to first route to the right paper, then zoom into the right section, then assemble evidence from exact spans.

## The latest result

The clearest way to see the effect of reranking is to put the two runs next to each other.

The first full evaluation was run on 100 queries with `k=5`, LLM judge enabled, and no reranker:

| Pipeline | Hit@1 | Recall@5 | MRR | NDCG@5 | Correctness | Faithfulness | Hallucination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical | 0.69 | 0.96 | 0.8012 | 0.8415 | 0.9300 | 0.7823 | 0.0278 |
| Vector | 0.60 | 0.89 | 0.7265 | 0.7682 | 0.9381 | 0.7862 | 0.0403 |
| Hybrid | 0.70 | 0.95 | 0.8078 | 0.8441 | 0.9428 | 0.7904 | 0.0426 |
| Zoom | 0.71 | 0.95 | 0.7944 | 0.8289 | 0.9495 | 0.8283 | 0.0034 |

Then I ran the same evaluation with `Alibaba-NLP/gte-reranker-modernbert-base` as a cross-encoder reranker over the top 30 candidates:

| Pipeline | Hit@1 | Recall@5 | MRR | NDCG@5 | Correctness | Faithfulness | Hallucination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical | 0.86 | 0.97 | 0.9017 | 0.9189 | 0.9360 | 0.7857 | 0.0456 |
| Vector | 0.85 | 0.96 | 0.8898 | 0.9073 | 0.9320 | 0.8007 | 0.0513 |
| Hybrid | 0.86 | 0.97 | 0.9003 | 0.9177 | 0.9433 | 0.7904 | 0.0386 |
| Zoom | 0.88 | 0.99 | 0.9253 | 0.9403 | 0.9600 | 0.8163 | 0.0170 |

This is the reranker lesson in one table pair. It does not merely polish the output. It changes what evidence reaches the answer model.

The biggest gains are in top-rank ordering. Lexical `Hit@1` moves from `0.69` to `0.86`; vector moves from `0.60` to `0.85`; hybrid moves from `0.70` to `0.86`; and Zoom moves from `0.71` to `0.88`. The same pattern appears in `MRR` and `NDCG@5`.

Zoom still has the best `Hit@1`, `Recall@5`, `MRR`, `NDCG@5`, judged correctness, judged faithfulness, and the lowest hallucination score. The advantage is not huge on every metric, but it is consistent across both retrieval-side and answer-side evaluation.

For the hallucination column, lower is better. The saved judge reasons make that interpretation clear: answers with `hallucination = 0` are often explicitly described as having no hallucinations. I updated the eval prompt to match this behavior, so the metric now means hallucination amount: `0` means no hallucination and `3` means severe hallucination before normalization.

That is exactly the kind of tradeoff I care about in an agent system. Retrieval is not only about whether the gold section appears somewhere in the top five. It is also about whether the answer model receives enough grounded evidence to answer without inventing details.

## Why Zoom improved

The first version of my Zoom eval had a bad agentic retry design. It asked an LLM whether the current evidence was enough. If the LLM said no, the pipeline rejected the current evidence and promoted other nodes from the reserve list.

That sounded reasonable, but it was wrong.

The LLM often rejected evidence that was actually from the right paper, because the visible snippet did not contain the exact answer. Once those nodes were rejected, the pipeline sometimes moved away from the right paper entirely. The retriever had done useful work, and the controller threw it away.

The fix was simple:

**Do not replace evidence. Add more evidence.**

Now the Zoom pipeline does one additional retrieve-more round. If the first five nodes are not enough, it keeps them and appends the next five ranked nodes from the reserve beam. The answer model then sees both the original evidence and the new evidence, within a configured context window.

This is closer to how a human reads a paper. If the first few passages are relevant but incomplete, you do not throw them away. You keep them open and read a little further.

That change matters because many RAG failures are not caused by completely wrong retrieval. They are caused by insufficient evidence. The answer may be one paragraph later, one neighboring section away, or hidden behind a title that only becomes meaningful with surrounding context.

This is also why Zoom has the lowest hallucination score in these runs.

The flat pipelines answer from a fixed top-k context. If the retrieved snippets are relevant but incomplete, the answer model may still try to bridge the gap. That is where small unsupported claims enter the answer.

Zoom reduces that failure mode in three ways. First, it keeps the initial evidence instead of replacing it, so useful partial context is not thrown away. Second, it can append more nearby or reserve evidence when the first set looks insufficient. Third, because the paper is represented as a hierarchy, the extra evidence usually comes from the same paper neighborhood instead of being a random global chunk. The answer model therefore sees more of the surrounding argument before it has to answer.

The lower hallucination score does not mean Zoom is intrinsically more truthful. It means this particular control loop gives the answer model less reason to guess.

## Query-aware snippets matter

Another fix was evidence clipping.

Previously, if a section was too long, the pipeline clipped from the beginning of the node. That is a common RAG mistake. It works when the answer is near the start, but research papers often put definitions, caveats, or experimental conclusions later in a section.

The updated pipeline clips long nodes around query-term matches. If the question asks about a specific method or concept, the snippet is centered near the best lexical match instead of blindly taking the prefix.

This sounds small, but it changes the job of the judge and the answer model. The model no longer has to infer that a broad section is probably useful. It is more likely to see the actual sentence that supports the answer.

I also split long paper sections into smaller child leaves while preserving the original section mapping for evaluation. This gives Zoom more precise retrieval units without losing the structured paper hierarchy.

## The reranker lesson

The rerank run confirms that a strong second-stage model can push retrieval metrics much closer to the ceiling.

This is a bigger deal than people often assume.

Most RAG discussions focus on the first-stage retriever: BM25 versus embeddings, vector database choice, chunk size, hybrid fusion, and so on. Those choices matter, but once the right evidence is somewhere in the candidate set, the reranker often becomes the component that decides whether the system succeeds.

A first-stage retriever is mostly a recall machine. It should cheaply gather plausible candidates. A reranker is a precision machine. It decides what the model actually sees.

That distinction matters:

- A vector retriever can find semantically related but non-answering chunks.
- A lexical retriever can find exact terms but miss the best explanatory passage.
- A hybrid retriever can increase recall but still order evidence poorly.
- A hierarchical retriever can route to the right paper but still need help choosing the exact span.

The reranker cleans up this last mile.

I added support for rerankers such as `Alibaba-NLP/gte-reranker-modernbert-base`, with a larger max input length and the same reranker interface wired into lexical, vector, hybrid, and Zoom. In this run, reranking is not an optional polish step. For research-paper QA, it is one of the most important pieces of the whole stack.

In other words:

**Good retrieval gets the answer into the room. A good reranker decides whether it gets a seat at the table.**

## What the metrics say

The rerank run shows several useful patterns.

First, lexical retrieval is still very strong. With reranking, it reached `Hit@1 = 0.86` and `Recall@5 = 0.97`, slightly ahead of vector retrieval on both metrics. This matches my experience from production systems: if the query contains exact terminology, symbols, method names, or dataset names, lexical retrieval is hard to beat.

This result is also a property of the benchmark. Open RAGBench over Arxiv papers is not a stress test for vague consumer intent. It is closer to a high-quality technical search problem. The documents are coherent, sectioned, and terminology-heavy. In that world, lexical matching gets a lot of signal for free. A vector model may understand that two passages are topically related, but topical relatedness is not always enough when the gold evidence is a specific section.

Second, vector retrieval became much more competitive after reranking. Its overall `Hit@1` rose to `0.85`, and its abstractive `Recall@5` reached `1.0`. But it still trailed lexical and hybrid on extractive top-1 accuracy. This is a good reminder that embeddings are not a universal upgrade. They change the error profile.

Third, hybrid retrieval was a solid middle ground, but not automatically dominant. It matched lexical on `Hit@1` and `Recall@5`, had the best answer correctness among the flat baselines, and reduced hallucination compared with lexical and vector. But Zoom still produced better judged answers.

Fourth, Zoom's request-more-docs behavior improved both retrieval-side and answer-side metrics. It was slower, because the assessment and retrieve-more loop add LLM calls, but it produced the highest retrieval scores and the lowest hallucination score in this run.

That latency tradeoff is real. Zoom averaged about 11.7 seconds of retrieval time and 16.0 seconds total time, while flat lexical retrieval averaged about 2.2 seconds of retrieval time and 4.7 seconds total time. This is not free quality. It is a choice: use the cheap flat retriever when the task is simple, and use structured retrieve-more behavior when answer reliability matters more than latency.

## Flat RAG versus agentic retrieval

The useful comparison here is not "RAG versus agents" in the abstract. All four systems are RAG systems. They retrieve evidence, pass it to an answer model, and evaluate the answer against a reference.

The difference is where control enters the pipeline.

The flat lexical, vector, and hybrid pipelines follow a fixed path:

- retrieve candidates,
- rerank candidates,
- take the top-k snippets,
- answer.

That design is fast, simple, and surprisingly strong on academic papers. With reranking, the flat baselines are good enough that I would not replace them by default. For many production questions, a flat hybrid pipeline with a good reranker is the right starting point.

Zoom adds a small agentic loop, but it does not make the whole retrieval process free-form. The deterministic parts still do most of the work: lexical ranking, vector scoring, hierarchy traversal, query-aware clipping, and reranking. The model is used at a narrower boundary: deciding whether the current evidence is sufficient, and asking for more context when it is not.

That distinction matters. The agentic part is not valuable because it "thinks" about the entire corpus. It is valuable because it changes the failure mode. A flat pipeline must bet that the top-k snippets are enough. Zoom can keep the first evidence set and append more evidence when the answer is under-supported.

In this benchmark, that behavior shows up most clearly in hallucination. Flat pipelines with reranking are strong, but their hallucination scores are still higher: lexical `0.0456`, vector `0.0513`, and hybrid `0.0386`. Zoom is slower, but its hallucination score is `0.0170`.

So my current view is:

- Use flat RAG when latency, simplicity, and cost dominate.
- Add a reranker before adding agentic behavior.
- Use agentic retrieve-more behavior when the main risk is answering from incomplete evidence.
- Keep the agentic loop narrow, inspectable, and grounded in deterministic retrieval steps.

## Takeaway

My earlier posts argued that agent systems are mostly ordinary software systems with LLM calls inside them. This evaluation reinforced that view.

The biggest improvements did not come from making the agent more autonomous. They came from fixing concrete systems problems:

- bad retry semantics,
- poor evidence clipping,
- over-broad sections,
- missing reranker wiring,
- and weak evaluation parsing.

The Zoom pipeline works better when it behaves less like a dramatic agent and more like a careful reader: start with likely evidence, keep it, request more when needed, rerank the candidates, and only then answer.

That is the pattern I trust most right now.

For practical RAG systems, I would summarize the lesson this way:

**Do not obsess over one retriever. Build a pipeline that can preserve recall, rerank precisely, and ask for more context before it guesses.**

---
title: 'Model Compression - Practical Methods for Local Deployment'
date: 2025-07-24
permalink: /posts/2025/07/LLMPruner/
tags:
  - LLM compression
  - AI on edge
---


Large Language Models (LLMs) like GPT, LLaMA, and their variants are increasingly used in real-world applications, but their size and computational demands make them difficult to deploy on consumer-grade hardware. To address this, **LLM compression** techniques aim to reduce memory usage, accelerate inference, and lower energy consumption—often with minimal impact on performance.

Broadly, compression methods fall into several categories:

* **Pruning** – removes redundant weights or structures
* **Low-Rank Factorization** – approximates large matrices with smaller ones
* **Quantization** – reduces weight/activation precision
* **Knowledge Distillation** – trains a smaller model to mimic a large one
* **Layer Replacement/Simplification** – substitutes complex components with lightweight modules
* **Weight Sharing** – reuses parameters across layers
* **Mixture-of-Experts Pruning** – sparsifies large MoE architectures
* **Token/Sequence Compression** – optimizes inputs and intermediate outputs

However, not all of these are practical for deployment on **consumer-level PCs**. Some methods—such as **knowledge distillation**, **sparsity-based unstructured pruning**, and **Mixture-of-Experts pruning**—require **additional training**, **custom inference engines**, or **complex infrastructure** to be effective. These approaches, while powerful in data center settings, are **less suitable for low-resource environments** due to:

* The need for large-scale retraining
* Poor hardware support for sparse matrix operations
* Inference inefficiency without specialized compilers or hardware


### 👉 Focus of This Article

In this article, we focus on **practical, post-training compression methods** that:

* Do **not require retraining from scratch**
* Can be applied to **existing pretrained models**
* Are compatible with **standard hardware (CPUs and GPUs)**
* Enable local deployment with **minimal engineering overhead**

Specifically, we cover:

* **Structured Pruning**
* **Low-Rank Factorization**
* **Post-Training Quantization**
* **Transformer Layer Removal or Simplification**

These techniques strike a balance between performance, simplicity, and deployability—making them ideal for developers seeking to run LLMs on local machines or edge devices.


## Pruning

Pruning is a model compression technique in machine learning that removes parts of a trained model deemed unnecessary for accurate predictions. The goal is to reduce model size, speed up inference, and lower computational costs—often with minimal loss in performance.

The core idea is simple: not all parameters contribute equally to a model’s output. By identifying and removing those with little influence, we can streamline the model while preserving its predictive power.

In traditional machine learning, pruning is commonly applied to decision trees using techniques like pre-pruning (early stopping) and post-pruning (trimming after training). In deep learning, pruning targets weights, neurons, or entire layers in overparameterized neural networks to enhance efficiency.

Modern deep models—especially Large Language Models (LLMs) like GPT-4 or LLaMA—often have billions or trillions of parameters. While this overparameterization boosts flexibility, it also results in:

1. High memory usage

2. Slower inference

3. Greater energy consumption

4. Deployment challenges on edge devices or real-time systems

As we do not train LLMs ourselves, we aim to obtain a compressed version of the weights provided by the open-source community—through post-training pruning—along with a small set of high-quality calibration data, which is essential for post-training quantization and pruning to recover the performance of compressed LLMs. Two main types of pruning are used in LLMs:

**Unstructured Pruning**: Removes individual weights (e.g., those near zero), creating sparse matrices that require specialized libraries for acceleration.

**Structured Pruning**: Removes entire neurons, attention heads, or layers, preserving dense matrix structures that are more hardware-friendly and easier to deploy.

* ### Loss-based Pruning

Loss-based pruning methods evaluate the importance of model components—such as weights, neurons, or layers—by estimating how much the training loss would increase if those components were removed. Most approaches use approximations based on first- or second-order derivatives of the training loss (e.g., gradients or Hessians). In some cases, especially for large language models, this may be computed using a small held-out dataset with a task-agnostic objective like masked language modeling or autoregressive loss.

The origins of loss-based pruning trace back to:

[OBD](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=17c0a7de3c17d31f79589d245852b57d083d386e) (Optimal Brain Damage) and [OBS](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) (Optimal Brain Surgeon) (LeCun et al., 1990; Hassibi & Stork, 1993), which estimate the increase in training loss using Taylor series approximations. OBD uses a diagonal approximation of the Hessian, while OBS considers the full Hessian for more accurate (but computationally expensive) pruning.

[Wanda](https://openreview.net/pdf?id=PxoFut3dWW) (Pruning by Weights and activations), although originally designed for unstructured and semi-structured pruning, introduces a weight-normalized activation-based metric. It estimates importance by multiplying each weight’s magnitude by the mean activation—offering a lightweight, loss-aware heuristic that can also be adapted to structured pruning by aggregating across larger units.

Modern structured pruning methods extend these ideas:

[LLM-Pruner](https://openreview.net/forum?id=J8Ajf9WfXP) identifies structural dependencies in LLMs and scores groups of parameters based on gradient information to perform one-shot width pruning. It does not neglect the first-order term in the Taylor expansion. This is important because the loss is computed on a dataset that the model has not been trained on, so gradients are non-zero and informative.
It approximates the diagonal of the Hessian using the Fisher Information Matrix, which is more tractable for large models and captures second-order curvature using squared gradients. It restores performance using LoRA.

[Shortened LLaMA](https://openreview.net/forum?id=18VGxuOdpu) applies depth pruning by removing entire Transformer blocks. It calculates block scores using Talyor series with only first-order derivatives of the loss on calib dataset, also a Perplexity variant to assess block importance and similarly relies on LoRA for recovery.

These approaches reflect the evolution from classic loss-sensitive pruning toward scalable, structure-aware methods suitable for compressing massive LLMs—while still grounded in the principle of minimizing loss impact.

* ### Magnitude-based Pruning

Magnitude-based methods use heuristics such as the norm or variance of weights to estimate importance. Simpler and computationally cheaper than loss-based methods, they assess structural salience by directly analyzing parameter values.

[FLAP](https://doi.org/10.1609/AAAI.V38I10.28960) introduces a structured fluctuation metric to identify unimportant weight matrix columns based on input variation, using a bias compensation mechanism for post-pruning recovery without fine-tuning. It requires no gradient or loss computation, uses bias compensation to recover performance without fine-tuning, and supports global compression via adaptive structure search. But heuristic-based importance may miss subtler interactions; compression quality can depend heavily on the input distribution used during pruning.

[SliceGPT](https://openreview.net/forum?id=vXxardq6db) applies PCA layer-wise to capture key signal directions, pruning columns or rows with low variance in the principal component space—preserving computational invariance in Transformer networks. The PCA-based metric provides a data-driven, statistically grounded importance estimate, preserves Transformer architecture invariants, and enables structured, hardware-friendly pruning.
while it can be computationally intensive for very large models; the approach assumes that low-variance components are always less informative, which may not hold for all tasks or domains.

* ### Regularization-based Pruning

Regularization-based methods incorporate sparsity-inducing penalties (adds a regularization term, such as L0, L1 and L2 regularization) into the loss function to encourage pruning during training. [Sheared LLaMA](https://openreview.net/forum?id=09iOdaeOzp) formulates pruning as a constrained optimization task using Lagrangian multipliers and pruning masks. It dynamically adjusts data loading across domains to improve training efficiency and guide pruning more effectively.

## Matrix Factorization

Low-rank factorization is a classic and intuitive technique for compressing large models by approximating weight matrices with the product of two smaller matrices. The basic idea is to exploit the inherent redundancy in overparameterized models by truncating the low-energy components—typically using singular value decomposition (SVD)—to reduce both memory and compute costs without heavily sacrificing performance. In its simplest form, a weight matrix $W \in \mathbb{R}^{m \times n}$ is factorized as $W \approx U V$, where $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{k \times n}$, with $k \ll \min(m, n)$.

Despite its long history in areas like signal processing and image compression, SVD-based methods have only recently gained traction in the context of LLM compression. Early efforts often suffered from significant accuracy degradation after truncating weights, requiring expensive fine-tuning to restore performance. To address this, recent works have introduced smarter, activation-aware strategies. For example, [ASVD](https://arxiv.org/abs/2312.05821) proposed scaling weight matrices with a diagonal matrix aligned with activation statistics to reduce the mismatch in output distributions. [SVD-LLM](https://arxiv.org/abs/2403.07378) further introduced a data whitening mechanism to preserve essential singular values that contribute to activation quality.

Building on these ideas, [Dobi-SVD](https://arxiv.org/abs/2502.02723) explores the use of differentiable truncation and weight remapping to adaptively select the rank for each layer based on model feedback. It also leverages incremental PCA for efficient weight updates and introduces a bijective mapping between rank and compression ratio to overcome limitations in conventional SVD truncation. These design choices enable Dobi-SVD to achieve high compression ratios with minimal performance degradation, all without requiring fine-tuning, making it a strong candidate for deployment on local hardware.

In summary, low-rank factorization—especially with recent innovations like ASVD, SVD-LLM, and Dobi-SVD—has evolved into a powerful and flexible approach for LLM compression, offering a promising balance between simplicity, efficiency, and performance.


## Architecture Simplification

Another effective compression strategy focuses on simplifying model architecture, particularly by reducing the depth of transformer-based LLMs. Instead of pruning individual weights or neurons, these methods remove entire transformer blocks or replace them with lightweight alternatives—making them highly compatible with standard hardware and efficient to deploy. Unlike unstructured pruning, these approaches maintain dense tensor operations and typically require only minimal post-pruning adaptation.

One of the most straightforward approaches is depth pruning, where full transformer layers are removed based on their contribution to model performance. Layers deemed less important are pruned in a one-shot fashion, and the resulting model is optionally fine-tuned using LoRA (like Shortened LLaMA mentioned above), continual pretraining, or both to restore any lost accuracy. Similarly, [ShortGPT](https://arxiv.org/abs/2403.03853) introduces the Block Influence (BI) metric, which uses cosine distance between hidden states before and after each layer to quantify its impact. Layers with low influence scores are removed, and retraining is recommended when high precision is required.

Building on this, [PruneMe](https://github.com/arcee-ai/PruneMe) takes a coarser-grained approach by evaluating fixed-length sequences of transformer layers rather than individual ones. The method measures cosine similarity between the input and output of these sequences, removing them entirely if the difference falls below a predefined threshold. To recover performance, LoRA-based tuning is applied selectively to the model’s MLP components.

Going beyond pruning, [LLM-Streamline](https://arxiv.org/abs/2403.19135) proposes to not only remove sequences of layers but to replace them with lightweight modules such as shallow FFNs or smaller transformer blocks. These replacement networks are trained using a combination of mean squared error (MSE) loss and standard LLM loss, with LoRA adapters enabling fast adaptation. This replacement approach preserves most of the original model’s behavior while significantly reducing depth and computational load.

[ReplaceMe](https://arxiv.org/abs/2505.02819) introduces a training-free, depth-wise pruning technique that replaces multiple transformer layers with a single linear transformation. After identifying layers with minimal contribution to performance by cosine distance (though L2 distance offers a closed form solution), ReplaceMe computes an optimal linear transformation that approximates the pruned layers’ functionality and integrates it into the preceding layer—without adding new parameters. The method also explores regularization strategies to balance perplexity and performance, and supports extensions with multiple transformations for more flexible pruning. This approach offers a simple yet effective path to compress LLMs without retraining, making it well-suited for deployment on resource-constrained devices.

These practical approaches are supported by an expanding line of scientific inquiry into how information is distributed across LLM layers. 
There is ongoing debate about how information is organized and used within large language models. Some studies suggest that factual knowledge is stored locally in MLP layers as key-value pairs and then propagated by self-attention through the network. Evidence supporting this view includes the ability to trace and edit specific factual associations and the phenomenon of "early exiting," where intermediate representations can be used to produce accurate outputs directly. However, other findings point to a more distributed storage pattern, showing that modifying facts often requires edits across multiple layers, especially when dealing with overlapping or relational information.
Meanwhile, methods like the [Tuned Lens](https://arxiv.org/abs/2303.08112) reveal that token prediction distributions converge in deeper layers, hinting at diminishing marginal utility from additional depth and supporting the idea that late-layer pruning is viable.

Even more compelling are observations from [Voita et al.](https://arxiv.org/abs/2309.04827) and [Liu et al.](https://arxiv.org/abs/2310.17157), who show a sparsity transition in activations around the network’s midpoint, and [Panigrahi et al.](https://arxiv.org/abs/2302.06600), who find that mid-layer weights change most during fine-tuning. These findings align with practical evidence: pruning only the deepest layers often has limited impact on performance, but beyond a certain point—roughly halfway—model quality degrades rapidly. This sharp pruning threshold suggests that while some late layers are redundant, the lower and middle sections of the network remain essential for preserving LLM capabilities.

Together, these studies and techniques make a compelling case for architecture simplification as a robust and interpretable compression method—one that aligns well with both theoretical understanding and real-world deployment needs.

## Quantization

Quantization is a model compression technique that reduces the numerical precision of weights and/or activations, typically from 32-bit floating-point (FP32) to lower bit-width formats such as FP16, INT8, or INT4. This enables more efficient model inference in terms of memory, latency, and energy usage, particularly important for deploying large models on edge devices or CPUs. While quantization has been widely adopted in computer vision, its application to large language models (LLMs) poses greater challenges due to the scale and sensitivity of transformer-based architectures.

**Quantization-Aware Training** (QAT) retrains quantized models to recover performance lost due to quantization noise. Although QAT significantly improves low-bit model performance, its application to LLMs is computationally demanding due to the scale of training. To alleviate this, recent works have explored integrating **Parameter-Efficient Fine-Tuning** (PEFT) techniques into QAT. For example, [QLoRA](https://arxiv.org/abs/2305.14314) combine quantization with adapters or LoRA to reduce training cost—though these methods are often task-specific. Overall, incorporating PEFT into QAT presents a promising direction for scalable and effective quantization of LLMs.

Recent work has focused on **post-training quantization** (PTQ) strategies that preserve model accuracy while reducing memory and compute costs. Here are some famous quant methods:
[GPTQ](https://arxiv.org/abs/2210.17323) proposes a Hessian-aware, layer-wise PTQ method that greedily quantizes weights to minimize output reconstruction error. By using a second-order approximation of the loss landscape, GPTQ achieves state-of-the-art accuracy at low bit-widths (e.g., INT3–4), though it requires significant calibration memory and runtime.
[AWQ](https://arxiv.org/abs/2306.00978) introduces an activation-aware quantization scheme that identifies and scales outlier channels in the weight matrix. It performs layer-wise INT4 quantization with minimal calibration and achieves performance close to GPTQ while being faster and more hardware-friendly.
[SmoothQuant](https://arxiv.org/abs/2211.10438) addresses the difficulty of quantizing both weights and activations by introducing a smoothing transformation that balances their dynamic ranges. This preprocessing step enables full INT8 quantization (including activations) and is effective for transformer-based models when a small calibration set is available.

These methods primarily target transformer blocks and focus on weight-only or weight+activation quantization under minimal architectural changes. Each presents a different trade-off between calibration cost, implementation complexity, and achievable accuracy.


### 👉 Quantization in `llama.cpp`: Practical Methods and Trade-offs

[llama.cpp]() implements several families of weight-only quantization schemes optimized for efficient inference on CPUs and constrained environments. These methods include legacy quants, K-quants, and I-quants, each offering trade-offs between performance, size, and accuracy.

**Legacy quantization formats** such as Q4\_0, Q4\_1, Q5\_0, and Q8\_0 are straightforward schemes that quantize weights in fixed blocks of 256, storing either a single scale (\_0 variants) or both scale and offset (\_1 variants) per block. These formats are simple to decode and perform well on older hardware but are generally outclassed in both speed and accuracy by more recent methods.

**K-quants** (e.g., Q3\_K\_S, Q4\_K\_M, Q5\_K\_XS), introduced in llama.cpp [PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684), improve on legacy formats through smarter bit allocation and mixed-bit quantization strategies. They maintain efficient decoding while assigning more precision to sensitive layers or blocks. Variants like `_S` or `_M` denote different profiles for bit distribution. K-quants typically outperform legacy quants in both accuracy and runtime across most hardware and are the default recommendation for general use.

**I-quants** (e.g., IQ2\_XXS, IQ3\_S, IQ4\_M), introduced in [PR #4773](https://github.com/ggerganov/llama.cpp/pull/4773), represent a state-of-the-art approach optimized for extremely low bits-per-weight. Inspired by external methods like QuIP#, they incorporate lookup tables and more complex decoding logic to preserve accuracy with fewer bits. While they offer the best compression-to-quality trade-off at 2–3 bits-per-weight, their runtime performance is lower due to higher decoding overhead, particularly on CPUs with limited SIMD capabilities.

An important orthogonal enhancement is the **importance matrix** ("imatrix"), which guides quantization to better preserve important weights. Though often introduced alongside I-quants, the imatrix is not tied to any particular format and can be applied to legacy and K-quants as well. It consistently improves quality, especially in low-bit settings, and is recommended regardless of the quant type, though its presence is not always reflected in model file names.

In practice, K-quants strike the best balance between performance and quality, while I-quants are preferable when extreme compression is needed and hardware constraints allow for their more demanding decoding process. Legacy quants remain relevant primarily for compatibility and performance on older systems.


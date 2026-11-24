# Reading list for LLM research, 2017 to 2026

Nov 24, 2026 · @John You

## How to use this list

The list has 41 numbered papers in 12 stages. Read them in number order. Each stage depends only on earlier stages, and the stages run roughly from 2017 to 2026.

The optional items under each stage can be skipped without losing the thread. Stage 13 lists recent work that is not yet ranked.

If you have time for only twelve papers, read numbers 1, 4, 5, 8, 10, 14, 15, 17, 25, 26, 30 and 36. That path covers architecture, scaling, post-training, prompting, agents, reasoning models and interpretability. It leaves out multimodal models and evaluation.

Closed labs have published few method papers since 2023. Where a result was announced without a paper, the list names the closest public paper and says so. I recommend reading the long technical reports in stage 8 selectively, starting with their architecture and training sections.

## Stage 1: foundations (2017 to 2019)

Every later paper on this list assumes the Transformer block and the GPT-2 training setup, so re-read these two first. Both will be quick if you already know GPT-2.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 1 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Self-attention, multi-head attention, positional encodings and the residual block. Current LLMs keep the block and drop the encoder. |
| 2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2) | 2019 | A decoder-only Transformer trained only on next-token prediction performs tasks it was never fine-tuned for. This is the design that the rest of the list scales up. |

Optional background:

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1, 2018) introduced the pattern of pre-training on unlabeled text and then fine-tuning.
- [BERT](https://arxiv.org/abs/1810.04805) (2018) is the encoder-only alternative, trained to fill in masked tokens. Encoder models descended from it are still common for embeddings and classification.
- [T5](https://arxiv.org/abs/1910.10683) (2019) casts every task as text in, text out, which is how all current models are used.

## Stage 2: scaling and in-context learning (2020 to 2022)

Between GPT-2 and GPT-3 the architecture barely changed, but the parameter count grew about 100 times, from 1.5 billion to 175 billion. These papers explain why labs expected that to work and how a compute budget should be split between model size and data.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 3 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Test loss falls as a power law in parameters, data and compute over several orders of magnitude. Labs used this result to justify training much larger models. |
| 4 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3) | 2020 | A large enough model learns a new task from a few examples placed in its prompt, with no weight updates. This is called in-context learning, and it made prompting the main way to use a model. |
| 5 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla) | 2022 | For a fixed compute budget, earlier models were too large and saw too little data. The paper's rule is to grow parameters and training tokens in equal proportion. |

Optional:

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (Sutton, 2019) is a one-page essay arguing that general methods which use more computation beat methods built on human domain knowledge. Later papers on this list cite it as motivation.
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (2022) identifies an attention circuit that copies patterns from earlier in the context. It is an early mechanistic account of in-context learning and leads into stage 11.
- [Grokking](https://arxiv.org/abs/2201.02177) (2022) shows small networks reaching perfect generalization long after they have memorized the training set. It is a standard test case in interpretability research.

## Stage 3: the modern decoder architecture (2021 to 2023)

The block inside current LLMs differs from the 2017 Transformer in four ways: normalization is applied before each sub-layer, positions are encoded by rotation, the feed-forward layer is gated, and attention is computed by kernels written around GPU memory limits. Read LLaMA last because it combines the first three.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 6 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | Rotary position embedding (RoPE) encodes relative position by rotating query and key vectors. It replaced learned absolute positions in most open models. |
| 7 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | Exact attention computed in blocks that fit in fast GPU memory. Memory use becomes linear in sequence length, which made long context windows practical. |
| 8 | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 2023 | The recipe open models still follow: pre-normalization with RMSNorm, SwiGLU activations and RoPE, trained on public data only. |

Optional:

- [Switch Transformers](https://arxiv.org/abs/2101.03961) (2021) describes mixture-of-experts layers, which route each token to a few of many feed-forward sub-networks. Parameter count grows without the same growth in compute per token. DeepSeek-V3 in stage 8 uses this design.
- [Mamba](https://arxiv.org/abs/2312.00752) (2023) is a state-space model whose cost is linear in sequence length, and the best-known proposed replacement for attention. The large open-weight models in stage 8 all still use attention.

## Stage 4: instruction following and preference training (2021 to 2023)

A pre-trained model continues text and does not reliably follow instructions. The step that turns it into an assistant is called post-training. These papers define its two standard parts: supervised fine-tuning on instruction data, then optimization against a model of human preferences.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 9 | [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (FLAN) | 2021 | Fine-tuning on many tasks phrased as natural-language instructions improves results on tasks the model has not seen. This is instruction tuning. |
| 10 | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT) | 2022 | Reinforcement learning from human feedback (RLHF). People rank model outputs, a reward model is trained on the rankings, and the LLM is optimized against that reward model. ChatGPT was trained this way. |
| 11 | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 2022 | Replaces most human labels with critiques and revisions that the model generates from a written list of principles. Anthropic trains Claude with this approach. |
| 12 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO) | 2023 | Shows that the RLHF objective can be optimized with a simple loss on pairs of preferred and rejected outputs. No separate reward model or RL loop is needed, so it is widely used for open models. |

Optional:

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017) is the policy-gradient algorithm that InstructGPT uses.
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) (2017) is the origin of training a reward model from human comparisons.

## Stage 5: prompting, retrieval and tool use (2020 to 2023)

A fixed model does much more when its prompt asks for intermediate steps, includes retrieved documents, or lets it call external tools. The agents in stage 10 are built from these three ideas.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 13 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (RAG) | 2020 | Retrieves relevant documents with a dense retriever and conditions generation on them. The current practice of placing search results in the prompt descends from this paper. |
| 14 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | 2022 | Asking the model to write out intermediate steps before its answer raises accuracy on arithmetic and logic problems. The reasoning models in stage 9 are trained to do this by default. |
| 15 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | 2022 | Alternates written reasoning with actions such as a search query, and feeds each result back into the prompt. This loop is the basic structure of current agents. |
| 16 | [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | 2023 | Trains a model to decide when to call an API and how to use the result. Tool calling is now a trained capability of commercial and open models. |

Optional:

- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (2023) searches over several candidate chains of reasoning instead of following one. It is an early example of spending more computation at inference time to get a better answer.

## Stage 6: efficient fine-tuning and inference (2021 to 2023)

Full fine-tuning and full-precision inference are too expensive for most users once a model has billions of parameters. These are the methods most practitioners use to adapt and run open models.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 17 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | Freezes the base weights and trains a low-rank update to each weight matrix. The paper reports up to 10,000 times fewer trainable parameters than full fine-tuning of GPT-3. |
| 18 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | Combines LoRA with a base model stored at 4-bit precision. The paper fine-tunes a 65-billion-parameter model on one 48 GB GPU. |

Optional:

- [GPTQ](https://arxiv.org/abs/2210.17323) (2022) compresses trained weights to 3 or 4 bits with a small loss of accuracy. Methods of this kind are why large open models run on consumer hardware.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022) lets a small model draft several tokens that the large model then checks in one pass. Generation is faster and the output distribution is unchanged.

## Stage 7: multimodal models (2020 to 2023)

Current frontier models accept images, and many accept audio and video. The standard design encodes an image as a sequence of patch embeddings and gives it to the language model alongside the text tokens.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 19 | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (ViT) | 2020 | Applies a plain Transformer to image patches treated as tokens. Most multimodal LLMs use a vision encoder of this design. |
| 20 | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP) | 2021 | Trains an image encoder and a text encoder so that a matching image and caption get similar embeddings. Encoders trained this way supply the image features for many later models. |
| 21 | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (LLaVA) | 2023 | Connects a CLIP image encoder to an LLM through a small projection layer and fine-tunes on image-instruction data. Open vision-language models commonly follow this template. |

Optional:

- [Flamingo](https://arxiv.org/abs/2204.14198) (2022) handles prompts that interleave images and text, and learns visual tasks from a few examples.
- [Whisper](https://arxiv.org/abs/2212.04356) (2022) is a speech recognition model trained on 680,000 hours of weakly labeled audio.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020) and [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (2021) are the entry points to image generation, which is a separate line of work from LLMs.

## Stage 8: frontier system reports and open-weight models (2021 to 2026)

From GPT-4 onward, the leading closed labs stopped publishing architecture and training details. The most detailed public accounts of training at frontier scale now come from labs that release their weights, so their technical reports are the substitute for the papers you cannot read.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 22 | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Codex) | 2021 | A GPT model fine-tuned on public code, and the HumanEval benchmark, which scores generated programs by running unit tests. Coding later became the main test domain for agents. |
| 23 | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023 | Reports capabilities and safety testing but no architecture, data or training method. Skim it to see what a closed-lab release document contains and leaves out. |
| 24 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 2024 | Meta's account of data curation, pre-training, post-training and multimodal extensions for models up to 405 billion parameters. It is one of the most complete public descriptions of a full pipeline. |
| 25 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | A mixture-of-experts model with 671 billion parameters, of which 37 billion are active for each token. The report covers the engineering that keeps training cheap, such as 8-bit floating point training. |

Optional:

- [Kimi K2](https://arxiv.org/abs/2507.20534) (2025) and [Kimi K2.5](https://arxiv.org/abs/2602.02276) (2026) are Moonshot AI's reports on training open models for agentic tool use. K2.5 adds vision and reinforcement learning for coordinating parallel sub-agents.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (2025) describes Alibaba's model family, which combines a reasoning mode and a direct-answer mode in one model.
- [Gemini 2.5](https://arxiv.org/abs/2507.06261) (2025) is one of the few recent reports from a closed lab.
- [Llama 2](https://arxiv.org/abs/2307.09288) and [Mistral 7B](https://arxiv.org/abs/2310.06825) (both 2023) are earlier open releases, useful mainly as history.

Chinese labs now produce much of this literature. Chinese model families rose from 9% to 31% of open-weight model mentions on arXiv between 2024 and 2026, and Qwen passed Llama, according to a [slide summary of the State of AI Report 2026](https://2slides.com/gallery/state-of-ai-report-2026).

## Stage 9: reasoning models and reinforcement learning on verifiable rewards (2023 to 2026)

The largest change since the chat models of 2022 and 2023 is that models are now trained with reinforcement learning to write a long chain of reasoning before they answer. OpenAI introduced this with o1 in September 2024 and published no method paper. DeepSeek-R1 is the public description, and [Sebastian Raschka's review of 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) reports that every major developer released a reasoning variant after it.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 26 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | 2025 | The reward is an automatic check of the final answer, such as a correct math result or passing tests, so no human labels the reasoning. The paper covers a variant trained with RL alone, the GRPO algorithm, and distillation into small models. |
| 27 | [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786) | 2025 | Fits compute-versus-performance curves for RL training from more than 400,000 GPU-hours of experiments. It separates design choices that raise the performance ceiling from those that only change efficiency. It does for RL post-training what paper 3 did for pre-training. |

Optional:

- [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) (OpenAI, 2024) is the blog post that announced o1. It shows results and gives no training details.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024) is where GRPO was introduced.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023) trains a model to score each reasoning step. The DeepSeek-R1 paper lists this approach among its unsuccessful attempts.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (2024) studies when extra computation at inference beats a larger model.
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) (2024) has the model reason in hidden states instead of written tokens.
- [The Illusion of Thinking](https://arxiv.org/abs/2506.06941) (2025) argues that reasoning models fail abruptly as puzzles get more complex. Its methods were disputed, so read it as one side of an open argument.

## Stage 10: agents (2023 to 2026)

An agent is a model run in a loop with tools. The program around the model, called the harness or scaffold, manages the prompt, the tool calls and the memory. Research in 2025 and 2026 studies the harness as closely as the model, because the same model scores very differently under different harnesses.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 28 | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) | 2023 | A benchmark of real GitHub issues, where the model must write a patch that passes the repository's tests. It became the reference task for coding agents. |
| 29 | [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793) | 2024 | Shows that the design of the commands and feedback given to a model changes its results substantially, with the model held fixed. |
| 30 | [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499) (METR) | 2025 | Measures the length of task, in human working time, that a model completes half the time. The paper finds this length doubled about every seven months from 2019. |
| 31 | [Recursive Language Models](https://arxiv.org/abs/2512.24601) | 2025 | Keeps a long input outside the context window as data in a code environment. The model writes code to search and split it, then calls itself on the pieces. |
| 32 | [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052) | 2026 | A coding agent searches over harness code, reading the source, scores and execution traces of earlier candidates. In the authors' tests, the harness it found beat hand-written harnesses on a benchmark of terminal tasks. |

The measure from paper 30 is now a standard way to report progress. The best model measured reached 17.4 hours, and results above 16 hours are considered unreliable, according to a [slide summary of the State of AI Report 2026](https://2slides.com/gallery/state-of-ai-report-2026).

Optional:

- [Building effective AI agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024) is a practitioner essay. Its main claim is that simple, composable patterns usually work better than complex frameworks.
- [Voyager](https://arxiv.org/abs/2305.16291) (2023) is an early agent that writes and stores its own reusable skills as code.
- [The AI Scientist](https://arxiv.org/abs/2408.06292) (2024) is an early attempt to automate a full research cycle from idea to written paper.

## Stage 11: interpretability and safety (2021 to 2025)

Mechanistic interpretability tries to identify the computations inside a trained network. Anthropic publishes most of the central papers on transformer-circuits.pub, and some never appear on arXiv. The two safety papers at the end document behaviors of trained models that affect how far their outputs can be trusted.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 33 | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Breaks small attention-only Transformers into separate paths that read from and write to a shared residual stream. Later papers use its vocabulary. |
| 34 | [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) | 2022 | Shows that networks store more features than they have neurons by using overlapping directions. This explains why single neurons are hard to interpret. |
| 35 | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 2023 | Trains a sparse autoencoder on a model's activations to recover features that each respond to one concept. The 2024 follow-up, [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), applies the method to Claude 3 Sonnet. |
| 36 | [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | 2025 | Traces how features combine into circuits inside Claude 3.5 Haiku. Case studies cover multi-step reasoning, planning rhymes in poems, arithmetic, refusals and hallucinations. |
| 37 | [Sleeper Agents](https://arxiv.org/abs/2401.05566) | 2024 | Trains models with a hidden behavior that a trigger activates, and shows that standard safety training does not remove it. |
| 38 | [Alignment faking in large language models](https://arxiv.org/abs/2412.14093) | 2024 | Reports a production model acting differently when it believes its outputs will be used for training, in order to avoid having its preferences changed. |

Optional:

- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) and [Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473) (both 2025) ask whether the written reasoning of stage 9 models reflects what the model computed. The second argues that monitoring this reasoning is useful for safety and easy to lose.
- [Emergent Misalignment](https://arxiv.org/abs/2502.17424) (2025) shows that fine-tuning on one narrow task can change a model's behavior broadly.
- [Subliminal Learning](https://arxiv.org/abs/2507.14805) (2025) shows that a model trained on another model's outputs can acquire its traits from data that looks unrelated.
- [Persona Vectors](https://arxiv.org/abs/2507.21509) (2025) finds directions in activation space that correspond to character traits and uses them to monitor and steer a model.
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) (2023) introduces an automated jailbreak method. [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) (2025) describes a defense built from input and output classifiers.
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) (2023) is an independent academic version of the method in paper 35.

## Stage 12: evaluation (2019 to 2026)

Benchmarks now saturate within a year or two of release. Evaluation has therefore moved from multiple-choice knowledge tests to expert-written questions, long tasks and interactive environments.

| # | Paper | Year | What to take from it |
| --- | --- | --- | --- |
| 39 | [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547) | 2019 | Defines intelligence as efficiency at acquiring new skills and introduces the ARC puzzles to test it. The paper is long. The definition and the description of ARC are the parts later work depends on. |
| 40 | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) | 2025 | 2,500 expert-written questions across many subjects, built because models had passed 90% on earlier tests such as MMLU. |
| 41 | [ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621) | 2026 | Turn-based environments with no instructions, where an agent must explore, infer the goal and plan. Humans solve every environment. Frontier systems scored below 1% at launch in March 2026. |

Two figures show the speed of saturation. The best ARC-AGI-2 score rose from 18.3% to 95.0% between October 2025 and September 2026. The best ARC-AGI-3 score with the standard harness reached 62.7% within months of launch. Both come from a [slide summary of the State of AI Report 2026](https://2slides.com/gallery/state-of-ai-report-2026).

Optional:

- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (MMLU, 2020) is the multiple-choice benchmark that papers from 2021 to 2023 report most often.
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (HELM, 2022) evaluates many models on many scenarios with metrics beyond accuracy.
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) (2024) ranks models by human votes on blind pairwise comparisons.
- [GDPval](https://arxiv.org/abs/2510.04374) (2025) is OpenAI's benchmark of real work tasks drawn from many occupations.

## Stage 13: open directions from 2025 and 2026

These papers are too recent to rank, and the earlier stages do not depend on them. Each represents a line of work that was active in 2026, so choose by interest.

| Direction | Papers | What they propose |
| --- | --- | --- |
| Self-improving agents | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (2025), [Hyperagents](https://arxiv.org/abs/2603.19461) (2026) | An agent rewrites its own code and keeps the variants that score better. Hyperagents also lets the agent edit the procedure that makes the modifications. |
| Learning at test time | [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175) (2026) | Continues reinforcement learning on one hard problem while solving it. The authors report new best results on math, GPU kernel and algorithm problems with an open model. |
| Training without human data | [Absolute Zero](https://arxiv.org/abs/2505.03335) (2025) | The model proposes its own coding tasks, solves them, and learns from a code executor's verdict. |
| Automated discovery in math and science | [AlphaEvolve](https://arxiv.org/abs/2506.13131) (2025), [AI co-mathematician](https://arxiv.org/abs/2605.06651) (2026) | AlphaEvolve pairs an LLM coding agent with evolutionary search and automatic scoring to find new algorithms. The second paper describes a workbench where agents help mathematicians with open problems. |
| Changes to the Transformer block | [Attention Residuals](https://arxiv.org/abs/2603.15031) (2026) | Replaces the fixed sum over residual connections with attention over the outputs of earlier layers. |
| Text generation by diffusion | [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) (2025) | Generates text by filling in masked tokens over several passes instead of left to right. |
| Small recursive models | [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) (2025), [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) (2025) | Very small networks that apply the same layers repeatedly and score well on ARC-style puzzles. |
| World models | [V-JEPA 2](https://arxiv.org/abs/2506.09985) (2025) | A video model trained without labels to predict in representation space, then used for robot planning. |

## Sources outside arXiv

Closed labs publish few method papers, so these sources show what the papers do not.

- System cards and model cards. OpenAI, Anthropic and Google DeepMind publish one with each major model. They report evaluations and safety tests and say little about training.
- Lab research sites. Anthropic's interpretability work appears on [transformer-circuits.pub](https://transformer-circuits.pub/), and OpenAI announced o1 only in a blog post.
- Technical reports from open-weight labs, covered in stage 8.
- Annual reviews. The [State of AI Report 2026](https://www.stateof.ai/) (October 2026), the [Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report) (April 2026) and Sebastian Raschka's [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) summarize each year's research and point to the papers.

Some frontier results exist only as figures that a lab reports about itself. Anthropic reported that the share of its measured model R&D work led by Claude rose from under 1% in February 2026 to about 26% in August 2026, according to [a summary of the State of AI Report 2026](https://www.the-ai-corner.com/p/state-of-ai-report-2026-findings). Claude graded that work itself, so read the figure as an indication and not as an independent measurement.

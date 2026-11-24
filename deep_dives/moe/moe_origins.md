## Question

Was there a single paper that introduced the idea of "Mixture of Experts"?

## Answer

Yes—the concept of **Mixture of Experts (MoE)** traces back to a single landmark 1991 paper: **"Adaptive Mixtures of Local Experts"** by Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton (Jacobs et al., 1991).

While the authors had discussed early versions of modular competitive networks prior to publication, this paper formalized the core MoE architecture: a set of separate "expert" neural networks paired with a trainable **gating network** that decides which expert should handle a given input (Jacobs et al., 1991).

## What the 1991 Paper Introduced

Before this work, combining multiple neural networks typically meant averaging their predictions, which forced networks to cooperate and cancel out each other's residual errors (Jacobs et al., 1991). Jacobs et al. (1991) changed the training objective so that experts **compete** rather than cooperate:

* **Task decomposition:** Each expert network specializes in a distinct region of the input space without interfering with the weights of other experts (Jacobs et al., 1991).
* **Input-dependent gating:** A separate feedforward gating network examines each input and assigns mixing proportions, routing training updates primarily to the best-fitting expert (Jacobs et al., 1991).
* **Faster convergence:** Tested on a multispeaker vowel-recognition task, the MoE architecture learned roughly twice as fast as standard backpropagation networks and scaled smoothly as more experts were added (Jacobs et al., 1991).

---

## How 1991 MoE Became Modern Sparse MoE

Although Jacobs et al. (1991) invented the idea, their original system functioned as a single standalone model rather than a component inside a massive deep network. The architecture used in modern large language models (LLMs) relies on a **sparse MoE layer**, illustrated below. Here, a **router (gating network)** dispatches each input embedding to only a few feed-forward network (**FFN**) experts—shown as Top-2 routing—before combining their outputs via weighted summation.

---

![Modern sparse MoE layer with Top-2 routing](images/Screenshot%202026-11-24%20at%2011.36.08%E2%80%AFAM.png)

*Modern sparse MoE layer with Top-2 routing. Source: Generated with Google Gemini*

---

Getting from the original 1991 formulation to that modern sparse routing architecture took three major follow-up papers:

* 1991: Adaptive Mixtures of Local Experts
Jacobs, Jordan, Nowlan, and Hinton introduce the foundational MoE architecture with competing local experts and a trainable gating network (Jacobs et al., 1991).


* 2013: Deep Mixture of Experts
Eigen, Ranzato, and Sutskever shift MoE from being the entire model to stackable layers embedded inside deep neural networks, allowing each layer to learn its own gating decisions (Eigen et al., 2013).


* 2017: Sparsely-Gated MoE Layer
Shazeer et al.—including Geoffrey Hinton as a co-author 26 years after the original paper—introduce sparse gating so only a tiny subset of thousands of experts activates per example, unlocking massive parameter counts at low compute cost (Shazeer et al., 2017).


* 2021: Switch Transformers
Fedus, Zoph, and Shazeer simplify sparse routing to a single active expert per token (top-1 routing) inside Transformer models, stabilizing training and scaling MoE to trillion-parameter language models (Fedus et al., 2021).


**References**

Eigen, D., Ranzato, M., & Sutskever, I. (2013). Learning Factored Representations in a Deep Mixture of Experts. *arXiv*. [https://doi.org/10.48550/arxiv.1312.4314](https://doi.org/10.48550/arxiv.1312.4314)
Cited by: 608

Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *arXiv*. [https://doi.org/10.48550/arxiv.2101.03961](https://doi.org/10.48550/arxiv.2101.03961)
Cited by: 7032

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, *3*, 79–87. [https://doi.org/10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79)
Cited by: 9755

Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *arXiv*. [https://doi.org/10.48550/arxiv.1701.06538](https://doi.org/10.48550/arxiv.1701.06538)
Cited by: 8467

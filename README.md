<p align="center"> <img src="assets/Money_Jump.png" alt="Monkey Jump Logo" width="400"/> </p> <h1 align="center"> Monkey Jump : MoE-Style PEFT for Efficient Multi-Task Learning</h1> <p align="center"> <a href="#" target="_blank"> <img src="assets/web.png" alt="Project Website" height="36"> </a> &nbsp;&nbsp;&nbsp;&nbsp; <a href="https://github.com/Nusrat-Prottasha/MonkeyJump" target="_blank"> <img src="assets/github.png" alt="GitHub" height="36"> </a> &nbsp;&nbsp;&nbsp;&nbsp; <a href="#" target="_blank"> <img src="assets/arxiv.png" alt="arXiv Paper" height="36"> </a> </p>
<p align="center"> <img src="assets/method.png" alt="Monkey Jump Logo" width="900"/> </p>
---

## 🔷 Abstract

Monkey Jump is a parameter-efficient fine-tuning (PEFT) method that achieves Mixture-of-Experts (MoE)-style specialization without introducing any new trainable parameters. Traditional MoE-PEFT approaches improve expressivity through token-specific expert routing, but incur additional memory, training cost, and parameter overhead due to added routers and expert modules—undermining the core goals of PEFT.

Monkey Jump avoids these costs by treating the PEFT adapters already present in each Transformer block (e.g., query, key, value, up, and down projections) as implicit experts, and routes tokens among them using $k$-means clustering with EMA-updated centers. This routing mechanism is entirely gradient-free and introduces no learned parameters.

We provide theoretical analysis showing that token-wise routing improves expressivity by avoiding cancellation effects found in uniformly applied adapters. In comprehensive multi-task experiments across 14 text, 14 image, and 19 video benchmarks, Monkey Jump achieves competitive performance with MoE-PEFT methods while using 7–29× fewer trainable parameters, up to 48% lower memory, and 1.5–2× faster training. Monkey Jump is architecture-agnostic and can be applied to any adapter-based PEFT method.

---


---

## 🚀 Features

- 🔀 MoE-style routing without any trainable parameters
- 🧠 Token-wise and sentence-wise clustering-based routing
- ⚡ 1.5–2× faster training and inference
- 💾 Up to 48% GPU memory savings
- 🔧 Compatible with LoRA, AdaLoRA, LoRA-FA, Propulsion
- 🧪 Gradient-free token routing via $k$-means + EMA

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/MonkeyJump.git
cd MonkeyJump
pip install torch torchvision torchaudio
pip install transformers accelerate datasets peft
pip install scikit-learn tqdm numpy pandas
```

---

## 💻 Quick Start

```python
from transformers import AutoModelForCausalLM
from src.MJLoRA import apply_monkeyjump

model = AutoModelForCausalLM.from_pretrained("model_name")
model = apply_monkeyjump(
    model,
    blocks={"LlamaDecoderLayer": list(range(32))},
    linears=["q_proj", "k_proj", "v_proj", "o_proj"],
    shared_expert=["up_proj", "down_proj"],
    rank=2,
    alpha=16.0,
    temperature=1.0,
    ema_momentum=0.2,
    top_k=2,
    rep_mode="token",
)
```

### Initialize Router

```python
from src.kmneas import init_router_centers

init_router_centers(
    trainer,
    subset_size=4000,
    kmeans_iters=15,
    rep_mode="token",
)
```

---

## 🧪 Routing Modes

| Mode        | Description               |
|-------------|----------------------------|
| `token`     | Per-token routing          |
| `last`      | Uses last token only       |
| `mean`      | Mean of all tokens         |
| `prompt_end`| Token at prompt boundary   |

---

### 📈 Efficiency Analysis

<p align="center">
  <img src="assets/effi.png" width="800" alt="Efficiency Chart"/>
</p>

Monkey Jump (MJ) is designed to deliver **MoE-style specialization** without compromising the efficiency that PEFT methods are known for. To evaluate this, we use the `LLaVA-OneVision-Qwen2-0.5B` backbone with `rank=2`, applying MoE-style PEFT to attention projections (Q, K, V, O).

All methods are benchmarked under the **same environment** for fairness:

- **GPU**: NVIDIA H100 80GB  
- **Framework**: PyTorch + HuggingFace Transformers  
- **Batch Size**: 8  
- **Gradient Accumulation**: 2  

The figure above compares **MJ variants** and **MoE-PEFT baselines** across 6 key metrics.

---

### 🔢 Parameter Efficiency

MJ uses **significantly fewer trainable parameters**:

| Method         | Params (K) |
|----------------|------------|
| MJ-Propulsion  | **49**     |
| MixLoRA        | 364        |
| HydraLoRA      | 909        |
| MoELoRA        | 1,425      |
| MJ-LoRAFA      | 98         |
| MJ-LoRA        | 270        |

> 🔍 *Despite the lower trainable parameter count, the total model size remains nearly identical (~1,705MB), as MJ reuses existing adapters instead of adding new experts.*

---

### 💾 Memory Efficiency

MJ significantly reduces peak GPU memory usage:

| Method         | Peak Memory (GB) |
|----------------|------------------|
| MJ-Propulsion  | **12.0**         |
| MoEAdaLoRA     | 23.2             |
| MoELoRA        | 22.8             |
| MJ-AdaLoRA     | 15.4             |

> 💡 *MJ achieves memory savings of up to **48%**, thanks to top-k sparse routing that activates fewer branches per pass.*

---

### ⚡ Training Speed

MJ improves training throughput and duration:

| Method         | It/s | Train Time (min) |
|----------------|------|------------------|
| MJ-Propulsion  | **5.94** | **5.0**         |
| MoE-PEFT Avg.  | 3.02–3.83 | 7.7–9.4        |

> 🚀 *All MJ variants exceed 4.8 it/s, while no MoE-PEFT method exceeds 3.9 it/s.*

---

### 🧠 Inference Speed

MJ maintains high throughput during inference:

- MJ-Propulsion achieves **15.8 it/s** on SST-2
- MoE-PEFT ranges from **9.4 to 12.8 it/s**

> 📌 *On average, MJ variants achieve **10–25% higher** inference throughput.*
>
> 
## 🔬 Theoretical Insights

Monkey Jump (MJ) is not only parameter-efficient—it’s also grounded in solid theoretical guarantees. Below are two key results explaining why MJ works better than standard PEFT and MoE-PEFT variants.

---

### 📈 1. Token-wise Routing Increases Expressivity

In standard PEFT, all adapters are applied uniformly to all tokens. The aggregate update has limited expressivity due to cancellation effects:

```math
U^{\text{PEFT}} = \left( \sum_{e=1}^{E} \Delta W_e \right) H
```

This summed update may have lower rank than the union of all adapter subspaces.

In contrast, MJ routes tokens selectively:

```math
U^{\text{MJ}} = \left[ \Delta W_1 H_1 \; \cdots \; \Delta W_E H_E \right]
```

Where \( H_e \) contains the tokens routed to adapter \( e \). This increases the span of outputs:

```math
\mathrm{rank}(U^{\text{MJ}}) \geq \mathrm{rank}(U^{\text{PEFT}})
```

> ✅ **Key Insight**: By avoiding overlap and cancellation, MJ preserves the diversity of adapter transformations.

---

### 🧠 2. Last-Token Routing is Information-Theoretically Optimal

When performing sequence-wise routing, we need a single representation to determine adapter routing.

In causal Transformers, the **last token representation** \( h_T \) is optimal because it has attended to the entire input sequence:

```math
I(h_T; X) \geq I(h_t; X) \quad \forall t < T
```

```math
I(h_T; X) \geq I(\bar{h}; X), \quad \text{where } \bar{h} = \frac{1}{T} \sum_{t=1}^T h_t
```

> ✅ **Conclusion**: Last-token routing retains the most semantic content and is superior to mean/max pooling for causal models.

---

These results show that MJ delivers both **efficient** and **expressive** token adaptation, grounded in provable design choices.

## 📊 Benchmarks

- 14 Text tasks (GLUE, CSQA, etc.)
- 14 Image tasks (ImageNet, VQA, etc.)
- 19 Video tasks (Action, Reasoning, etc.)
- 🧠 Matches MoE-PEFT performance with fewer parameters and faster training

---

## 📜 Citation

```bibtex
@article{prottasha2025monkeyjump,
  title={MoE-Style PEFT for Efficient Multi-Task Learning},
  author={Prottasha, Nusrat Jahan and Kowsher, Md and Yu, Chun-Nam and Chen, Chen and Garibay, Ozlem},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## 📝 License

MIT License

---

## 📬 Contact

- GitHub: [Nusrat-Prottasha/MonkeyJump](https://github.com/Nusrat-Prottasha/MonkeyJump)
- Email: your.email@university.edu

---

**You've experienced ScholarGPT — now meet what's next.**  
*Scholar Deep Research Agent* elevates your research game with:  
🔍 350M+ trusted papers from top academic publishers, updated hourly.  
🧠 Advanced multiple AI models dig through millions of sources for pinpoint insights, fast.  
📝 Auto-generated highlights, smart notes, and visual reports  
📁 All saved directly to your AI-powered knowledge base  
ScholarGPT helped you search. Now, transform how you think.  
[Explore Scholar Deep Research](https://bit.ly/43rXgSx)

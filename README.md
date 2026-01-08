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

- 🔀 MoE-style routing with zero routing parameters
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

## 📊 Benchmarks

<p align="center">
  <img src="assets/effi%20(1).png" width="800" alt="Efficiency Chart"/>
</p>

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

# 🌟 XINGYUN LAB - Research Code Repository

Welcome to **XINGYUN LAB**'s official code repository! This repository contains open-source implementations of our published research papers.

---

## 📚 Publications

### 2026

#### ACL 2026
**[Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning](https://arxiv.org/abs/2601.07408)**

*Ziheng Li, Liu Kang, Feng Xiao, Luxi Xing, Qingyi Si, Zhuoran Li, Weikang Gong, Deqing Yang, Yanghua Xiao, Hongcheng Guo*

**Abstract:** We identify a critical limitation in Group Relative Policy Optimization (GRPO) for reasoning tasks: its coarse-grained credit assignment, which propagates group-level rewards uniformly to every token in a sequence, overlooking the varying contributions of individual steps. To address this, we propose Outcome-grounded Advantage Reshaping (OAR), a mechanism that reallocates advantage based on outcome-sensitivity. We introduce two strategies: OAR-P uses counterfactual token perturbations to estimate outcome sensitivity, while OAR-G employs input-gradient sensitivity as a proxy. These signals are combined with a conservative, dual-tier advantage reshaping scheme that suppresses low-impact tokens and boosts pivotal ones. Experiments show that OAR-P sets a performance ceiling, while OAR-G achieves comparable gains at negligible computational overhead, with both significantly outperforming GRPO baselines.

**Resources:**
- 📄 [Paper (arXiv)](https://arxiv.org/abs/2601.07408)
- 💻 [Code](./OAR-mathematical-reasoning/)
- 🏆 **Accepted at ACL 2026**

---

## 🗂️ Repository Structure

Each paper has its own directory containing:
- `README.md` - Detailed instructions and documentation
- `requirements.txt` - Python dependencies
- `code/` - Source code implementation
- `data/` - Sample datasets or data preprocessing scripts (if applicable)
- `experiments/` - Experiment configurations and scripts

```
XINGYUN_LAB/
├── README.md
├── OAR-mathematical-reasoning/
│   ├── README.md
│   ├── requirements.txt
│   ├── code/
└── [future papers...]
```

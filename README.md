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
- 💻 [Code](./OAR/)
- 🏆 **Accepted at ACL 2026**

---

## 🗂️ Repository Structure

```
XINGYUN_LAB/
├── README.md
├── OAR/
│   └── main.py
└── [future papers...]
```

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/XINGYUN-AI-LAB/XINGYUN_LAB.git
   cd XINGYUN_LAB
   ```

2. **Navigate to a specific paper**
   ```bash
   cd OAR
   ```

3. **Follow the instructions in each paper's directory**

---

## 👥 Team

**XINGYUN LAB** - AI and NLP Research Laboratory

---

## 📧 Contact

For questions about specific papers, please open an issue in this repository or contact the corresponding author listed in the paper.

---

## 📝 Citation

If you find our work useful, please cite the relevant paper:

```bibtex
@inproceedings{li2026outcome,
  title={Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning},
  author={Li, Ziheng and Kang, Liu and Xiao, Feng and Xing, Luxi and Si, Qingyi and Li, Zhuoran and Gong, Weikang and Yang, Deqing and Xiao, Yanghua and Guo, Hongcheng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

---

## 📜 License

Each project may have its own license. Please refer to the LICENSE file in each paper's directory.

---

**Last Updated:** April 2026

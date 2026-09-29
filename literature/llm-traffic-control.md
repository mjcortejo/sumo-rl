# Language models for traffic signal control

Related work for the paper. All entries were checked against their arXiv or publisher page on the date noted. None
of these uses a contrastive / ranking ("System One") model: all use generative LLMs, or LLMs as teachers or
coordinators for other controllers.

| Work | Authors | Date / venue | Approach | Why it matters to us |
| --- | --- | --- | --- | --- |
| LLMLight (arXiv:2312.16044) | Siqi Lai et al. | KDD 2025, DOI 10.1145/3690624.3709379 | LLM as the signal-control agent; trains LightGPT, a traffic-specialized LLM | Closest analogue to our base-vs-specialized comparison. Likely a key related work and possibly a baseline. |
| CoLLMLight (arXiv:2503.11739) | Zirui Yuan, Siqi Lai, Hao Liu | 2025-03-14, under review | Cooperative LLM agents for network-wide control | Multi-intersection extension (relevant to the 4x4 grid). |
| iLLM-TSC (arXiv:2407.06025) | not yet checked | 2024 | Combines RL and an LLM to refine the policy | Hybrid RL+LLM related work. |
| LATS (arXiv:2603.24361) | Yifeng Zhang, Peizhuo Li, Tingguang Zhou, Mingfeng Fan, Guillaume Sartoretti | 2026-03-25 | LLM teacher distils semantic features into a MARL student; no LLM at deployment | An alternative answer to the inference-cost problem that CLM also tackles. |
| LLM-Augmented TSC (arXiv:2604.23902) | Jiazhao Shi | 2026-04-26 | LSTM prediction plus LLM decision support with safety filter | SUMO, fixed-time and rule-based baselines, similar setup to ours. |
| ReasonLight (arXiv:2605.29425) | Aoyu Pang, Maonan Wang, Yuejiao Xie, Chung Shue Chen, Zhiwei Yang, Man-On Pun | 2026-05-28 | Multimodal foundation model plus RL for zero-shot handling of unseen events | Zero-shot generalization (our RQ4). |
| HiLLTS (arXiv:2607.22691) | Yue Ding, Tendai Mukande, Mingming Liu | 2026-07-17 | City-level LLM coordinates districts (normal / restrict / green wave), zero-shot | Uses fixed-time and max-pressure baselines like ours, and reports CO2. |
| ProcessLight (arXiv:2609.22746) | Huaitao Zhao, Tianlong Zhou, Weijie Wang, Jiasheng Shi, Weixiong Rao | 2026-09-19, accepted to EMNLP 2026 | Process supervision (STeP-PO) for step-level credit assignment in LLM reasoning for signal control | Newest state of the art for generative LLM controllers. |

Also seen but not yet verified: "LLM-enabled universal traffic signal control across different intersections and
traffic flows" (Knowledge-Based Systems, 2026, ScienceDirect S095070512600403X), a pretrained-LLM generalized TSC
framework in SUMO; and "Large Language Models (LLMs) as Traffic Control Systems at Urban Intersections: A New
Paradigm" (arXiv:2411.10869).

## Novelty check

As of 2026-09-29 no work applies a contrastive / System One model (CLM-8B, Jev) to traffic signal control. Our
novelty claim holds. Re-check at every `/literature-watch` run.

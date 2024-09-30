# I Need Help! Evaluating LLM’s Ability to Ask for Users’ Support: A Case Study on Text-to-SQL Generation
**TL;DR:** We propose a framework for LLMs to seek user support, design evaluation metrics to measure the trade-off between performance boost and user burden, and empirically assess this ability on Text-to-SQL generation.

**Paper link:** https://arxiv.org/abs/2407.14767

![Figure 1: Overview of our experiments on text-to-SQL. LLMs struggle to determine when they need help based solely on the instruction (x) or their output (y). They require external feedback, such as the execution results (r) from the database, to outperform random baselines.](assets/Figure_1.pdf)

## Playground
See `playground.ipynb` for step-by-step walkthrough of how to obtain "need-user-support probability" with toy examples.

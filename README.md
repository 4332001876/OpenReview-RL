# OpenReview-RL
> Warning: AI generated README, will be updated later

## Introduction
OpenReview-RL is a project that explores the application of Reinforcement Learning (RL) to improve large language models' performance on paper evaluation tasks. This project specifically focuses on training an AI model to review and rate academic papers similar to the review process on OpenReview.org.

## Motivation
While reinforcement learning has shown significant benefits for reasoning-based tasks, its impact on non-reasoning tasks like paper evaluation remains underexplored. This project aims to:
- Investigate how RL can enhance a model's ability to provide structured and consistent paper reviews
- Develop an evaluation framework for academic paper quality assessment
- Compare RL-fine-tuned models against traditional fine-tuning approaches for this specific domain

## Methodology
Our approach follows a training methodology similar to Deepseek R1:

1. **Task Definition**: Train models to rate paper abstracts on a scale of 1-10, providing both reasoning and a final numerical score
2. **Structured Output**: Enforce a strict output format with reasoning in `<think>` tags and the final rating in `<answer>` tags
3. **Reward Function**: Develop specialized reward functions that consider:
   - Format adherence (proper use of specified tags)
   - Rating accuracy (compared to ground truth scores)
   - Reasoning quality (evaluated through auxiliary models)

## Project Structure
```
OpenReview-RL/
├── examples/
│   └── data_preprocess/     # Data preprocessing utilities
├── verl/
│   └── utils/
│       └── reward_score/    # Reward functions for RL training
└── scripts/                 # Training and evaluation scripts
```

## Setup and Usage
1. **Installation**
```bash
git clone https://github.com/yourusername/OpenReview-RL.git
cd OpenReview-RL
pip install -e .
```

2. **Data Preparation**
```bash
python examples/data_preprocess/prepare_openreview_data.py
```

3. **Training**
```bash
python scripts/train_openreview_model.py
```

4. **Evaluation**
```bash
python scripts/evaluate_openreview_model.py
```

## Results
Our preliminary findings suggest that RL-based fine-tuning provides several benefits for paper evaluation tasks:
- Improved adherence to structured output formats
- More consistent numerical ratings
- Better correlation with human expert evaluations
- Enhanced reasoning capability when explaining rating decisions

Detailed experimental results and model comparisons will be provided in the associated technical report.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.


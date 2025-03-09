# HW5

## Baseline Guide

| Baseline | Public score | Estimated time(kaggle) |
| --- | --- | --- |
| Simple | 15.05 | 1h |
| Medium | 18.44 | 2h |
| Strong | 23.57 | 3h |
| Boss | 30.08 | >12h |

- Simple Baseline: Train a simple RNN seq2seq to acheive translation

    > BLEU: 

- Medium Baseline: Add learning rate scheduler and train longer
    
    > BLEU: 12.46

- Strong Baseline: Switch to Transformer and tuning hyperparameter

    > BLEU: 26.65

- Boss Baseline: Apply back-translation
   
    > BLEU: 19.61

## Gradescope Overview

- (2pts) Problem 1

    - Visualize the similarity between different pairs of positional embedding and briefly explain the result.

    - Additionally,attach the code that you used for visualization.
  
- (2pts) Problem 2

    - Clip gradient norm and visualize the changes of gradient norm in
different steps.Circle two places with gradient explosion.
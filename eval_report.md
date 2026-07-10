# Evaluation Report

Goals evaluated: 10

## Self-correcting agent

- Completion rate: 100%
- Average steps taken: 5.6
- Total self-corrections triggered: 6
- Runs with unresolved (recovery-failed) subtasks: 0

## Baseline (no self-correction)

- Completion rate: 80%
- Average steps taken: 5.0
- Total self-corrections triggered: 0 (baseline never self-corrects, by design)
- Runs with unresolved subtasks (tool just failed silently): 2

  Unresolved (no recovery attempted):
  - [f65e881c6341] Write a function that counts word frequency in a string and  -> unresolved: ['s5']
  - [216c31a50848] Write a function that computes the factorial of a non-negati -> unresolved: ['s5']

## Delta

- Completion rate improvement from self-correction: +20 percentage points
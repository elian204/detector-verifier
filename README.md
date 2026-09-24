# Detector as verifier

Train a small language model so that a frozen AI-text detector scores its
writing as human, and measure what that training actually changes.

The policy is `Qwen/Qwen3-0.6B-Base`. The reward is a frozen
`qwen3-variable` detector trained in
[rasbt/ai-detector-from-scratch](https://github.com/rasbt/ai-detector-from-scratch).
The trainer is that repository's stage-18 script, used as published:
group-relative REINFORCE, no policy-ratio clip, no KL penalty. For each
prompt the model samples several answers. An answer is paid

```text
reward = (1 - P_AI) × length_score
```

`P_AI` is the detector's probability that the answer is AI-written.
`length_score` is 1 at the requested word count and falls off when the
answer is shorter or longer. The update only ranks answers to the same
prompt.

A high training reward is not the result. The result is whether the text
stays readable and whether a detector that was not the reward agrees.

## Result so far

Short answers, 50 and 100 words, four samples per step, 500 steps.
Recorded on the fork, branch `feat/robustness-evaluation`:
[pilot write-up](https://github.com/elian204/ai-detector-from-scratch/blob/feat/robustness-evaluation/results/grpo-human-baseline/README.md).

The training detector saturates. By about step 50 its human score is
about 0.9996 and stays there for ordinary prose, repeated sentences, and
digit salad. Answers in a group tie on that score, so later updates follow
the length term. Float32 collapses to identical strings in 4 of 4 seeds.
Bfloat16 does not within 500 steps. In one bfloat16 run the final prose is
coherent, the training detector scores it near 0.9 human, and a held-out
detector scores it near 0.3.

Repetition and gibberish move automatic fluency scores in opposite
directions. Both the held-out detector and the text itself have to be
checked.

## In progress

A 250-word run, to see whether saturation and length-only updates persist
when the detector sees more text. Same trainer, same reward, float32,
learning rate `1e-5`, four samples per step, so the only change from the
short run is length. The trainer's own cap applies: about 416 new tokens
for a 250-word target. Checkpoints are compared with the short run at
steps 0, 20, 50, and 100. The same outputs are scored with DistilBERT,
which is not the reward.

If that run repeats the short-run failure, the next training change is a
KL penalty toward the frozen base model. If it does not move, the
advantages and token-cap rate come first. If the held-out score and the
text both improve, that is the stopping point.

Before any of those calls, a blind read of before/after samples checks
coherence and whether the answer addresses the prompt.

## Backlog

- Why bfloat16 collapsed less than float32: frozen weights, or a smaller
  effective step.
- Training with `--skip-zero-advantage-updates` turned off.
- A clipped policy ratio. It does nothing on one update of fresh samples,
  because the ratio starts at 1.
- Eight samples per prompt, a supervised warmup, or a logistic-regression
  reward.

Weights are not stored in this repository.

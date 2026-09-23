# Detector as verifier

A learning project on what happens when a small language model is trained,
with group-relative policy gradients, to look less like AI to a frozen
text detector.

The goal is to understand the training loop and to keep a public record of
each step. It is not to build an undetectable writer.

Upstream trainer and dataset construction:
[rasbt/ai-detector-from-scratch](https://github.com/rasbt/ai-detector-from-scratch).

The finished short-target pilot lives on the fork, branch
`feat/robustness-evaluation`:
[results write-up](https://github.com/elian204/ai-detector-from-scratch/blob/feat/robustness-evaluation/results/grpo-human-baseline/README.md).
That history stays there. This repository starts the next project.

## What the trainer does

Raschka's script is left unchanged. For one prompt it samples a few answers
and pays each one

```text
reward = (1 - P_AI) * length_score
```

`P_AI` is the frozen detector's probability that the answer is AI-written.
`length_score` is 1 when the word count matches the request, and lower when
the answer is too short or too long. The update compares answers to the same
prompt only. There is no value network, no policy-ratio clip, and no KL
penalty back to the base model. The loss is group-relative REINFORCE on the
sum of completion-token log probabilities.

## What pilot 1 already showed

Prompts asked for 50 or 100 words. Four answers were sampled per step, for
500 steps. The reward model was `qwen3-variable`.

By about step 50 the training detector's human score sits near 0.9996 and
stops moving. Inside a group, the answers tie on that score, so the update
is driven by length. The highest-reward text can be one sentence repeated
to the token cap. Float32 runs collapse to identical strings in 4 of 4
seeds. Bfloat16 runs do not, within 500 steps. One bfloat16 seed ends in
coherent prose that the training detector likes and a held-out detector
does not.

Repetition and gibberish move fluency metrics in opposite directions, so a
single automatic score is not enough. Samples have to be read.

## What we will do

Each step ends in a commit and a push. Model weights stay on disk.

1. **This commit.** The plan, and nothing else.
2. **Read pilot 1 in the text.** Annotate one high-reward repeated answer
   and one late collapsed or gibberish answer from the existing logs.
   No new training.
3. **250-word pilot.** Same trainer, same reward, float32, learning rate
   `1e-5`, **four** rollouts (the same group size as pilot 1). About 100
   steps, with checkpoints at 0, 20, 50, and 100 on a fixed set of
   validation prompts. Leave the trainer's token cap alone: a 250-word
   target is already limited to about 416 new tokens. Score the same
   outputs with DistilBERT, which was not the reward. The question is
   whether detector saturation and length-only updates still happen when
   the answer is long enough for the detector to see more text.
4. **Read that run before changing the algorithm.** Compare it with pilot 1
   at the same step counts. A small blind read of before/after samples
   judges coherence and whether the answer is on topic. Detector scores
   do not decide that.
5. **One change, and only if step 3 shows the same hack.** Add a KL penalty
   toward the frozen base model. If the 250-word run does not move, inspect
   advantages and token-cap hits before adding anything. If the held-out
   detector and the blind read both improve, stop and write that up.

KL tests whether staying near the base model preserves readable text under
this reward. It does not make a saturated detector able to tell prose from
garbage.

## Not next

These are real questions. They wait until step 5 has a result.

- Why bfloat16 collapsed less than float32 (frozen weights versus a smaller
  effective step).
- Turning off `--skip-zero-advantage-updates`.
- A ratio clip. On a single update of fresh samples the ratio starts at 1,
  so clipping does nothing until samples are reused.
- Eight rollouts, a supervised warmup, or a logistic-regression reward.
- Stacking more than one of these in the same run.

## How a run is judged

Report the training-detector score, the held-out score, the length term,
and a short read of the text. A rising training score by itself is not
success. Longer answers that only improve the length term are not reward
hacking. Hacking is a training score that stays high while the text gets
worse, or while a detector that was not the reward disagrees.

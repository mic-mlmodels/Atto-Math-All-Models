import math

MAX_LR = 2e-5
MIN_LR = 1e-6
NUM_STEPS = 5000 // 32
WARMUP_STEPS = NUM_STEPS // 10


def lr_scheduler(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return (MIN_LR / MAX_LR) + (1.0 - MIN_LR / MAX_LR) * 0.5 * (
        1.0
        + math.cos(
            math.pi * min(1.0, (step - WARMUP_STEPS) / max(1, NUM_STEPS - WARMUP_STEPS))
        )
    )

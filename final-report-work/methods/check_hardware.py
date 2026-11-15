"""Validate hardware draw invariants before the robustness study."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.hardware import draw_ordered_spacing

torch.manual_seed(42)
state = torch.get_rng_state().clone()
draws = []
for seed in [3101, 3102, 3103]:
    d, attempt = draw_ordered_spacing(0.005, 64, seed)
    repeat, repeat_attempt = draw_ordered_spacing(0.005, 64, seed)
    assert torch.equal(d, repeat) and attempt == repeat_attempt
    assert torch.equal(state, torch.get_rng_state())
    positions = d.flatten() * torch.arange(-31.5, 32.)
    assert bool((positions.diff() > 0).all()) and bool((d > 0).all())
    draws.append(d)
    print(f'hardware seed {seed}: ordered, reproducible, accepted attempt {attempt}')
assert all(not torch.equal(draws[i], draws[j]) for i in range(3) for j in range(i))
print('PASS: independent draws, strict ordering and unchanged training random state')

"""Independent, reproducible ordered-array impairment draws."""
import torch


def draw_ordered_spacing(wavelength, antennas, seed, max_attempts=1000):
    """Use the reference lambda/25 spacing perturbation and alternating sort.

    A private CPU generator leaves training randomness untouched. Reject draws
    whose physical element positions are not strictly ordered; do not shrink
    or otherwise tune accepted impairments.
    """
    if antennas < 2 or antennas % 2:
        raise ValueError('This reference ordering requires an even antenna count')
    wavelength = float(wavelength)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(-(antennas-1)/2, (antennas-1)/2+1)
    for attempt in range(1, max_attempts + 1):
        raw = wavelength/2 + wavelength/25 * torch.randn(antennas, generator=generator)
        ordered = raw.sort().values
        spacing = torch.cat([ordered[::2].flip(0), ordered[1::2]])
        if bool((spacing > 0).all()) and bool((torch.diff(indices * spacing) > 0).all()):
            return spacing.reshape(-1, 1), attempt
    raise RuntimeError(f'No ordered impairment draw after {max_attempts} attempts')

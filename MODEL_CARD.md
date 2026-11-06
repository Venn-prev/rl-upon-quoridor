# Qudor AI — Champion Model Card

## Selected checkpoint

`checkpoints/gen85_best.pt` is the strongest published official-rules model in this
repository. It uses Gumbel AlphaZero search with a 128-channel, 10-block
Squeeze-and-Excitation ResNet and the v3 16-plane canonical encoder.

- Generation 85, iteration 1759; Quoridor rules version 2.
- Checkpoint size: 13,018,066 bytes.
- SHA-256: `2126e3c2937916972d633efee237f0176cc4e3b8bb23fb86de86189a8712e170`.
- The checkpoint contains champion weights, not the original optimiser or replay state.
  Training from it therefore creates a fresh training lineage.

## Evaluation evidence

The published CPU tournament used 16 search simulations per move:

| Opponent | Wins | Draws | Losses | Win rate |
|---|---:|---:|---:|---:|
| `marcobt15` | 270 | 0 | 0 | 100.0% |
| `dimitrijekaranfilovic` | 170 | 0 | 0 | 100.0% |
| `gorisanson` | 267 | 0 | 3 | 98.9% |
| `sigma` | 151 | 3 | 96 | 60.4% |

These are results against the listed engines, not an official human-rating benchmark.
They provide the available evidence for retaining gen85 over the older checkpoints.

## State and action spaces

- Observation: 16 planes of shape `(16, 9, 9)`, canonicalised for the player to move.
- Policy: 209 Quoridor actions (81 pawn destinations and 64 placements for each wall
  orientation).
- Value: scalar in `[-1, 1]`.
- Rules version 2 enforces wall geometry, pawn jumps, and path preservation.

## Training profile

`configs/rtx3060_6gb.json` retains the champion's network and Gumbel search, while using
one self-play worker, 24 concurrent games, a batch of 128, mixed-precision training, and
disk-backed replay for a 6 GB RTX 3060. The output directory defaults to
`runs/rtx3060_6gb` on the D: drive within the repository.

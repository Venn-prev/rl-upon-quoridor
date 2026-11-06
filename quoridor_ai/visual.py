"""Board visualization for Quoridor - pygame (interactive) and matplotlib (static).

Usage:
    from quoridor_ai.core.engine import State, legal_actions, apply
    from quoridor_ai.visual import show_pygame, show_matplotlib

    s = State()
    show_matplotlib(s, "board.png")          # save a PNG, no GUI needed
    show_pygame(s)                           # interactive window, click to play

Coordinates follow the engine: cell index i -> (r, c) = divmod(i, 9),
row 0 is player 1's start side, row 8 is player 0's goal.
"""
from __future__ import annotations

import math

from .core.engine import State, legal_actions, apply

N = 9
CELL = 64          # pixel size of one cell (pygame)
WALL = 10          # wall thickness in pixels
MARGIN = 40

# colors
BG = (24, 26, 33)
CELL_BG = (44, 48, 60)
GRID = (90, 96, 110)
P0 = (232, 82, 82)     # red   - starts row 8, goal row 0
P1 = (82, 148, 232)    # blue  - starts row 0, goal row 8
WALL_C = (210, 190, 120)
LEGAL = (110, 200, 110)
TEXT = (220, 220, 225)


def _walls(s: State):
    """Yield ('h'|'v', r, c) for every wall slot occupied in state s."""
    for r in range(8):
        for c in range(8):
            b = 1 << (r * 8 + c)
            if s.h & b:
                yield "h", r, c
            if s.v & b:
                yield "v", r, c


# ---------------------------------------------------------------- matplotlib
def show_matplotlib(s: State, path: str | None = None, probs=None):
    """Render the board with matplotlib. Saves to `path` if given,
    otherwise opens an interactive window. `probs` (optional, len 81 or 209)
    shades each cell by policy probability."""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    if probs is not None and len(probs) == 209:
        pv = np.zeros(81)
        for i in range(81):
            pv[i] = probs[i]
        tot = pv.sum()
        probs = pv / tot if tot > 0 else None

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#181a21")
    ax.set_facecolor("#181a21")
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_ylim(N - 0.4, -0.6)          # row 0 on top
    ax.set_aspect("equal")
    ax.axis("off")

    # cells
    for r in range(N):
        for c in range(N):
            fc = "#2c303c"
            if probs is not None:
                p = float(probs[r * 9 + c])
                fc = matplotlib.colors.to_hex(
                    plt.cm.Greens(min(1.0, p * 6)))
            ax.add_patch(mpatches.Rectangle((c - .5, r - .5), 1, 1,
                                            facecolor=fc, edgecolor="#5a606e",
                                            linewidth=1.2))
    # goals
    for c in range(N):
        ax.plot(c, -0.42, marker="^", color=P0, markersize=6)
        ax.plot(c, N - 0.58, marker="v", color=P1, markersize=6)

    # walls
    for o, r, c in _walls(s):
        if o == "h":
            ax.plot([c - .5, c + 1.5], [r + .5, r + .5],
                    color="#d2be78", linewidth=7, solid_capstyle="butt")
        else:
            ax.plot([c + .5, c + .5], [r - .5, r + 1.5],
                    color="#d2be78", linewidth=7, solid_capstyle="butt")

    # pawns
    r0, c0 = divmod(s.p0, 9)
    r1, c1 = divmod(s.p1, 9)
    ax.add_patch(mpatches.Circle((c0, r0), 0.32, facecolor=P0,
                                 edgecolor="white", linewidth=2, zorder=5))
    ax.add_patch(mpatches.Circle((c1, r1), 0.32, facecolor=P1,
                                 edgecolor="white", linewidth=2, zorder=5))

    ax.set_title(
        f"ply {s.ply}   P0 walls {s.walls0}   P1 walls {s.walls1}   "
        f"turn: {'P0' if s.player == 0 else 'P1'}",
        color=TEXT, fontsize=11)

    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"saved {path}")
    else:
        plt.show()
    return fig


# ------------------------------------------------------------------- pygame
def show_pygame(start: State | None = None, model=None, sims: int = 64,
                cell: int = CELL, human_player: int | None = 0):
    """Interactive pygame window. Click a highlighted cell to move,
    press H/V then click a slot edge to place a wall (or use mouse hover).
    Close the window or press ESC to quit."""
    import pygame

    if start is None:
        start = State()

    W = H = MARGIN * 2 + cell * N
    board_x = MARGIN
    board_y = MARGIN
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Quoridor")
    font = pygame.font.SysFont("consolas", 16)
    big = pygame.font.SysFont("consolas", 22, bold=True)
    clock = pygame.time.Clock()

    def to_cell(mx, my):
        c = (mx - board_x) // cell
        r = (my - board_y) // cell
        if 0 <= r < N and 0 <= c < N:
            return int(r) * 9 + int(c)
        return None

    def cx(cell_i):     # cell index -> pixel center
        r, c = divmod(cell_i, 9)
        return board_x + c * cell + cell // 2, board_y + r * cell + cell // 2

    def wall_rect(o, r, c):
        x = board_x + c * cell
        y = board_y + r * cell
        if o == "h":
            return pygame.Rect(x, y + cell - WALL // 2, 2 * cell, WALL)
        return pygame.Rect(x + cell - WALL // 2, y, WALL, 2 * cell)

    def wall_from_click(mx, my, orient):
        # nearest slot: walls sit on the grid line at the *bottom/right* of slot (r,c)
        gx = (mx - board_x) / cell
        gy = (my - board_y) / cell
        if orient == "h":
            c = int(math.floor(gx - 0.5))
            r = int(math.floor(gy - 0.5))
        else:
            c = int(math.floor(gx - 0.5))
            r = int(math.floor(gy - 0.5))
        if 0 <= r < 8 and 0 <= c < 8:
            return r * 8 + c
        return None

    state = start
    pending_wall = None       # None | 'h' | 'v'
    legal = legal_actions(state)
    probs = None

    def net_probs():
        nonlocal probs
        if model is None:
            probs = None
            return
        try:
            import torch
            from .core.encoding import encode_v1
            x = torch.from_numpy(encode_v1(state)).unsqueeze(0)
            with torch.no_grad():
                p, _ = model(x)
            probs = p.squeeze(0).numpy()
        except Exception:
            probs = None

    net_probs()

    def refresh():
        nonlocal legal, pending_wall
        legal = legal_actions(state)
        pending_wall = None
        net_probs()

    def ai_move():
        nonlocal state, legal, probs
        if model is None or state.winner is not None:
            return
        try:
            from .az_selfplay import search  # noqa: F401  (optional hook)
        except Exception:
            pass
        # simple greedy: pick argmax policy among legal actions
        if probs is not None and len(probs) == 209:
            best = max(legal, key=lambda a: probs[a])
            state = apply(state, best)
            refresh()

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        hover_cell = to_cell(*mouse)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_h:
                    pending_wall = "h" if pending_wall != "h" else None
                elif ev.key == pygame.K_v:
                    pending_wall = "v" if pending_wall != "v" else None
                elif ev.key == pygame.K_a:
                    ai_move()
                elif ev.key == pygame.K_r:
                    state = start.copy()
                    refresh()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if state.winner is not None:
                    continue
                if state.player != (human_player if human_player is not None
                                    else state.player) and model is not None:
                    continue
                if pending_wall:
                    slot = wall_from_click(*mouse, pending_wall)
                    if slot is not None:
                        a = (81 if pending_wall == "h" else 145) + slot
                        if a in legal:
                            state = apply(state, a)
                            refresh()
                    pending_wall = None
                else:
                    ci = hover_cell
                    if ci is not None and ci in legal:
                        state = apply(state, ci)
                        refresh()

        # ----- draw
        screen.fill(BG)
        # cells
        for r in range(N):
            for c in range(N):
                i = r * 9 + c
                col = CELL_BG
                if i in legal:
                    col = LEGAL
                elif probs is not None:
                    p = float(probs[i])
                    if p > 0.01:
                        col = (int(CELL_BG[0] + p * 120),
                               int(CELL_BG[1] + p * 160),
                               CELL_BG[2])
                pygame.draw.rect(screen, col,
                                 (board_x + c * cell + 1,
                                  board_y + r * cell + 1, cell - 2, cell - 2),
                                 border_radius=4)
        # goal arrows
        for c in range(N):
            pygame.draw.polygon(screen, P0,
                                [(board_x + c * cell + cell // 2, board_y - 6),
                                 (board_x + c * cell + cell // 2 - 6, board_y - 14),
                                 (board_x + c * cell + cell // 2 + 6, board_y - 14)])
            pygame.draw.polygon(screen, P1,
                                [(board_x + c * cell + cell // 2, board_y + N * cell + 6),
                                 (board_x + c * cell + cell // 2 - 6, board_y + N * cell + 14),
                                 (board_x + c * cell + cell // 2 + 6, board_y + N * cell + 14)])
        # walls
        for o, r, c in _walls(state):
            pygame.draw.rect(screen, WALL_C, wall_rect(o, r, c),
                             border_radius=3)
        # hover preview for wall placement
        if pending_wall:
            slot = wall_from_click(*mouse, pending_wall)
            if slot is not None:
                a = (81 if pending_wall == "h" else 145) + slot
                col = LEGAL if a in legal else (150, 60, 60)
                rect = wall_rect(pending_wall, slot // 8, slot % 8)
                pygame.draw.rect(screen, col, rect, 3, border_radius=3)

        # pawns
        for pos, colr in ((state.p0, P0), (state.p1, P1)):
            x, y = cx(pos)
            pygame.draw.circle(screen, colr, (int(x), int(y)), cell // 3)
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)),
                               cell // 3, 2)

        # HUD
        turn = "P0 (red)" if state.player == 0 else "P1 (blue)"
        hud = f"ply {state.ply}   walls P0 {state.walls0} / P1 {state.walls1}   turn {turn}"
        screen.blit(font.render(hud, True, TEXT), (10, 6))
        mode = "wall: press H/V, click a slot  |  click cell = move" \
            if pending_wall is None else f"placing {pending_wall}-wall (ESC/H/V to cancel)"
        screen.blit(font.render(mode, True, (160, 165, 175)), (10, H - 26))
        if state.winner is not None:
            t = big.render(f"PLAYER {state.winner} WINS!  (R = restart)", True,
                           (255, 220, 90))
            screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 14))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

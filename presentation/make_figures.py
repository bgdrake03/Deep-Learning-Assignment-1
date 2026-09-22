"""Diagrams for the presentation deck.

    python presentation/make_figures.py

Writes three PNGs into presentation/figures. These are drawn for the talk
rather than measured from the data, so they live here and not in results/.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEEP = '#0B2E33'
TEAL = '#0E7C7B'
MINT = '#D6F5F3'
CORAL = '#C9502F'
MUTED = '#557575'

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x, y, w, h, label, sub=None, face=MINT, edge=TEAL, ink=DEEP,
        size=13, sub_size=11, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.07',
                                facecolor=face, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h * (0.63 if sub else 0.5), label, ha='center',
            va='center', fontsize=size, fontweight='bold', color=ink)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center',
                fontsize=sub_size, color=ink)


def arrow(ax, x1, y1, x2, y2, color=TEAL, lw=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=20, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.tight_layout(pad=0.2)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def contrast():
    """The usual way of training versus what we had to do."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis('off')

    ax.plot([6.5, 6.5], [0.45, 4.35], color='#D9E5E4', lw=2)

    # ---- left: load everything
    ax.text(3.1, 4.28, 'The usual way', ha='center', fontsize=15,
            fontweight='bold', color=DEEP)
    box(ax, 0.45, 2.25, 2.9, 1.5, '500,000 rows',
        'the whole file in memory', size=14)
    arrow(ax, 3.45, 3.0, 4.15, 3.0)
    box(ax, 4.25, 2.35, 1.7, 1.3, 'Model', face=DEEP, edge=DEEP, ink='white',
        size=14)
    ax.text(3.1, 1.55, 'Needs a computer big enough', ha='center', fontsize=13,
            color=CORAL, fontweight='bold')
    ax.text(3.1, 1.15, 'to hold the entire file at once', ha='center',
            fontsize=13, color=CORAL, fontweight='bold')

    # ---- right: one chunk at a time
    ax.text(9.9, 4.28, 'What we did', ha='center', fontsize=15,
            fontweight='bold', color=DEEP)
    chunk_y = [3.45, 2.78, 2.11, 1.44]
    labels = ['chunk 1', 'chunk 2', '. . .', 'chunk 10']
    # arrowheads land at separate points on the model's edge, so they do not
    # pile up into a blob
    for y, label, target in zip(chunk_y, labels, [3.36, 3.05, 2.74, 2.43]):
        box(ax, 6.95, y, 1.85, 0.56, label, size=12, lw=1.5)
        arrow(ax, 8.9, y + 0.28, 10.28, target, lw=1.4)
    box(ax, 10.35, 2.3, 1.8, 1.3, 'Model', face=DEEP, edge=DEEP, ink='white',
        size=14)
    ax.text(9.9, 0.95, 'Memory stays the same', ha='center', fontsize=13,
            color=TEAL, fontweight='bold')
    ax.text(9.9, 0.55, 'however big the file gets', ha='center', fontsize=13,
            color=TEAL, fontweight='bold')

    return save(fig, 'contrast.png')


def pipeline():
    """Three passes over the file, in plain language."""
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis('off')

    box(ax, 0.25, 1.95, 2.1, 1.35, 'pricing.csv', '500,000 rows',
        face=DEEP, edge=DEEP, ink='white', size=14)
    ax.text(1.3, 1.6, 'never fully opened', ha='center', fontsize=11,
            color=MUTED, style='italic')

    box(ax, 2.95, 1.95, 1.95, 1.35, 'Read a chunk', '50,000 rows', size=13)
    arrow(ax, 2.4, 2.62, 2.9, 2.62)

    rows = [
        (3.45, 'Pass 1  ·  Measure', 'what is typical, and how spread out'),
        (1.95, 'Pass 2  ·  Learn', 'update the model, 32 rows at a time'),
        (0.45, 'Pass 3  ·  Test', 'set aside rows the model never saw'),
    ]
    for y, label, sub in rows:
        box(ax, 5.5, y, 3.6, 1.1, label, sub, size=13, sub_size=10.5)
        arrow(ax, 4.95, 2.62, 5.45, y + 0.55)

    outs = [(3.45, 'the average and spread'),
            (1.95, '12,500 small updates'),
            (0.45, 'an honest score')]
    for y, label in outs:
        box(ax, 9.7, y, 3.05, 1.1, label, size=13)
        arrow(ax, 9.15, y + 0.55, 9.65, y + 0.55)

    ax.text(6.5, 4.62,
            'Only one chunk is ever in memory — the file could be any size',
            ha='center', fontsize=13.5, fontweight='bold', color=CORAL)

    return save(fig, 'pipeline.png')


def architecture():
    """The network, with the jargon kept to a minimum."""
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 4.3); ax.axis('off')

    layers = [
        (0.45, 'What we know', '5 facts', '', MINT),
        (2.9, 'Layer 1', '64 units', 'sigmoid', MINT),
        (5.1, 'Layer 2', '32 units', 'sigmoid', MINT),
        (7.3, 'Layer 3', '16 units', 'sigmoid', MINT),
        (9.5, 'Prediction', '1 number', 'linear', DEEP),
    ]

    for i, (x, name, units, act, face) in enumerate(layers):
        ink = 'white' if face == DEEP else DEEP
        box(ax, x, 1.7, 1.85, 1.5, name, units, face=face,
            edge=DEEP if face == DEEP else TEAL, ink=ink, size=13.5)
        if act:
            ax.text(x + 0.925, 1.42, act, ha='center', fontsize=11,
                    color=CORAL, fontweight='bold')
        if i:
            arrow(ax, x - 0.5, 2.45, x - 0.05, 2.45)

    ax.text(1.375, 3.5, 'sku, price, order,\nduration, category', ha='center',
            fontsize=11.5, color=MUTED, va='center')
    arrow(ax, 11.4, 2.45, 11.85, 2.45)
    ax.text(12.0, 2.45, 'units sold', fontsize=13.5, fontweight='bold',
            color=DEEP, va='center', ha='left')

    ax.text(6.4, 0.85,
            'Each layer looks for patterns in what the layer before it found.',
            ha='center', fontsize=13, color=DEEP)
    ax.text(6.4, 0.42,
            'Three hidden layers with sigmoid, as the assignment requires.',
            ha='center', fontsize=12.5, color=MUTED)

    return save(fig, 'architecture.png')


if __name__ == '__main__':
    for p in (contrast(), pipeline(), architecture()):
        print(p)

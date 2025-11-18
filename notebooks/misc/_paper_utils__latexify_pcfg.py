import random
from copy import deepcopy
from collections import defaultdict

from nltk import Nonterminal, Tree
from nltk.grammar import ProbabilisticProduction, PCFG
from nltk.tree.transforms import chomsky_normal_form

S, NP, VP, DT, JJ, NN, V, PP, IN = map(
    Nonterminal,
    ["S","NP","VP","DT","JJ","NN","V","PP","IN"]
)




# raw = [
#     (S,  [NP, VP],     1.0),
#     (NP, [DT, JJ, NN], 0.9),
#     (NP, [NP, PP],     0.1),
#     (VP, [V, PP],      0.8),
#     (VP, [VP, NP],     0.2),
#     (PP, [IN, NP],     1.0),
#     (DT, ["The"],    0.7),
#     (JJ, ["quick"],  0.6),
#     (NN, ["fox"],    0.4),
#     (V,  ["jumps"],  1.0),
#     (IN, ["over"],   1.0),
#     (DT, ["the"],    0.3),
#     (JJ, ["lazy"],   0.4),
#     (NN, ["dog"],    0.6),
# ]
raw = [
    (S,  [NP, VP],   1.0),
    (NP, [DT, JJ, NN], 1.0),
    (VP, [V, PP],    1.0),
    (PP, [IN, NP],   1.0),
    (DT, ["The"],    0.7),
    (JJ, ["quick"],  0.6),
    (NN, ["fox"],    0.4),
    (V,  ["jumps"],  1.0),
    (IN, ["over"],   1.0),
    (DT, ["the"],    0.3),
    (JJ, ["lazy"],   0.4),
    (NN, ["dog"],    0.6),
]
# raw = [
#     (S,  [NP, VP],     1.0),
#     (NP, [DT, JJ, NN], 0.9),
#     (NP, [NP, PP],     0.1),
#     (VP, [V, PP],      0.8),
#     (VP, [VP, NP],     0.2),
#     (PP, [IN, NP],     0.7),
#     (PP, ["VP"],       0.3),
#     (DT, ["The"],      1.0),
#     (JJ, ["quick"],    0.6),
#     (JJ, ["lazy"],     0.4),
#     (NN, ["fox"],      0.4),
#     (NN, ["dog"],      0.6),
#     (V,  ["jumps"],    1.0),
#     (IN, ["over"],     1.0),
# ]
productions = [
    ProbabilisticProduction(lhs, rhs, prob=p)
    for lhs, rhs, p in raw
]
grammar = PCFG(S, productions)


def sample_prob_tree(sym):
    prods = [p for p in grammar.productions() if p.lhs() == sym]
    probs = [p.prob() for p in prods]
    prod  = random.choices(prods, weights=probs, k=1)[0]

    children = []
    for X in prod.rhs():
        if isinstance(X, Nonterminal):
            children.append(sample_prob_tree(X))
        else:
            children.append(Tree(str(X), []))

    label = f"{sym}\\\\({prod.prob():.1f})"
    return Tree(label, children)

tree_nonbin = sample_prob_tree(grammar.start())

tree_bin = deepcopy(tree_nonbin)
chomsky_normal_form(tree_bin, factor="right", horzMarkov=2)

counters = defaultdict(int)
def rename_internals(t):
    lbl = t.label()
    if "|" in lbl:
        orig = lbl.split("|", 1)[0]
        base = orig.split("\\\\")[0]
        counters[base] += 1

        t.set_label(f"{base}_{counters[base]}\\\\(1.0)")
    for ch in t:
        if isinstance(ch, Tree):
            rename_internals(ch)

rename_internals(tree_bin)


def forest_format(t, indent=0):
    pad = "  " * indent

    if isinstance(t, str):
        return pad + t

    if all(isinstance(c, Tree) and len(c) == 0 for c in t):
        child = t[0].label()
        return f"{pad}[{t.label()}[{child}]]"

    t_label = t.label()
    if "_" in t_label:
        t_label = t_label.split("_")[0] + "\\_" + t_label.split("_")[1]
    s = f"{pad}[{t_label}\n"
    for c in t:
        s += forest_format(c, indent+1) + "\n"
    s += f"{pad}]"
    return s


print("% Non-binary PCFG parse (forest syntax):\n")
print(forest_format(tree_nonbin))
print("\n% Binarized (CNF) parse with NP_1, NP_2:\n")
print(forest_format(tree_bin))

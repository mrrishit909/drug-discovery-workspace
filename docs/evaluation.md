# Evaluation

Everything here is measured on synthetic libraries generated with seeds the demo does not use (`(101, 102, 103)`): the same cores, linkers and
substituents, a different hidden binding rule's noise and a different measured history. Reproduce with `python -m dd.evaluate`
(about three minutes). The numbers say how well the methods recover what the generator hid. They are not a claim about a real
target: the hidden rule is a sum of substructure effects with one interaction, which fingerprints were made to find, and a real
structure-activity landscape is far rougher.

## 1. The affinity model

Five gradient-boosted regressors on fingerprints and descriptors, trained on the 300 measured compounds, scored on every unmeasured
compound against the hidden truth (assay noise 0.3; the hidden rule's own noise 0.45, so about 0.45 is the floor).

| library (seed) | measured | RMSE | nearest-neighbour RMSE | inside the 90% band | uncertainty tracks error (rank corr.) | uncertainty scale applied |
|---|---|---|---|---|---|---|
| 101 | 300 | 0.637 | 0.853 | 83.1% | 0.132 | 3.36 |
| 102 | 300 | 0.629 | 0.868 | 88.0% | 0.099 | 4.22 |
| 103 | 300 | 0.618 | 0.852 | 90.7% | 0.105 | 3.81 |

The ensemble's raw spread was two to three times too narrow; one scale factor fitted on a held-out fifth brings the band to about
87.3%, still short of 90%. The uncertainty says little about which compounds will be wrong (rank correlation near
zero): it is wide where the training data is thin, not where the rule is hard.

## 2. Which 96 compounds for the next plate?

Three ways to fill a plate under a budget of 200 cost units (1 in stock, 5 to make), each scored on the hidden truth: actives found
(pIC50 at or above 7), distinct scaffolds, and the model's error on the rest of the library after the plate is folded in.

| library | selection | compounds | actives found | scaffolds | RMSE before | RMSE after |
|---|---|---|---|---|---|---|
| 101 | upper confidence + diversity | 84 | 76 | 7 | 0.632 | 0.672 |
| 101 | exploit (top predicted) | 76 | 76 | 1 | 0.638 | 0.654 |
| 101 | random | 96 | 7 | 9 | 0.637 | 0.621 |
| 102 | upper confidence + diversity | 80 | 77 | 5 | 0.629 | 0.626 |
| 102 | exploit (top predicted) | 88 | 88 | 5 | 0.63 | 0.621 |
| 102 | random | 96 | 11 | 8 | 0.629 | 0.61 |
| 103 | upper confidence + diversity | 96 | 67 | 6 | 0.616 | 0.59 |
| 103 | exploit (top predicted) | 84 | 84 | 6 | 0.618 | 0.609 |
| 103 | random | 96 | 7 | 9 | 0.618 | 0.588 |

Means: actives found, upper confidence + diversity: 73 / exploit (top predicted): 83 / random: 8; RMSE after the plate, upper confidence + diversity: 0.629 / exploit (top predicted): 0.628 / random: 0.606.

Exploitation finds the most actives and the model learns nothing from them: it fills the plate from series it already knows.
Random finds almost no actives and is the only selection that improves the model. Upper confidence with a diversity rule found
fewer actives than exploitation and taught the model no more, which is a worse result than the method's reputation: with a
rule this additive, the uncertain compounds it reaches for are expensive and uninformative. The honest conclusion is that on
this generator the choice is between actives now (exploit) and a better model (random), and the API exposes the knob (kappa)
rather than pretending to have resolved it.

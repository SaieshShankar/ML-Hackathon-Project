# PathSnap Phase 2

The selected CPU pipeline scored **0.8825 CV Macro F1**, then **0.9127 Macro F1** and **91.98% Top-1** on the 10 student-test trips. The student-test Macro F1 change was +0.62 percentage points against the matched untuned run. Tuning improved 5 of the 10 student-test trips.

| Split | Scorer | Decoder | Macro F1 | Top-1 | Candidate log loss | Direct continuity | Sequence similarity |
|---|---|---|---:|---:|---:|---:|---:|
| Training CV | Untuned | Independent | 0.8072 | 79.95% | 0.0961 | 93.88% | 0.7706 |
| Training CV | Untuned | Viterbi | 0.8805 | 82.40% | 0.0961 | 98.46% | 0.9332 |
| Training CV | Tuned | Independent | 0.8007 | 79.78% | 0.0978 | 93.64% | 0.7654 |
| Training CV | Tuned | Viterbi | 0.8825 | 82.32% | 0.0978 | 98.27% | 0.9286 |
| Student test | Untuned | Independent | 0.8155 | 88.90% | 0.0663 | 95.08% | 0.8002 |
| Student test | Untuned | Viterbi | 0.9065 | 92.84% | 0.0663 | 98.92% | 0.9549 |
| Student test | Tuned | Independent | 0.8019 | 88.59% | 0.0678 | 94.66% | 0.7914 |
| Student test | Tuned | Viterbi | 0.9127 | 91.98% | 0.0678 | 98.79% | 0.9544 |

## What ran

- Gaussian-process Bayesian optimisation used expected improvement for 8 LightGBM trials and 12 decoder trials, including baselines and initial designs. Each classifier trial fit five 64-trip training subsets and scored the corresponding 16-trip held-out folds. The objective was mean decoded observation-level Macro F1.
- Selected LightGBM settings were 340 trees, learning rate 0.045, 31 leaves, minimum 70 samples per leaf and L2 regularisation 3.0.
- The decoder used emission weight 0.7, learned transition weight 0.7, gap threshold 30 s, gap decay exponent 0.25 and extra bounded-unreachable penalty 0.0.
- Candidate arcs are the hidden states. LightGBM supplies log emission scores. Directed road connectivity and a learned transition scorer provide transition scores. The decoder resumes after unmatched observations, weakens transition costs after long GPS gaps, and emits a route break for choices without a path in the bounded graph search.
- Both classifiers were refit on the same 80 training trips after freezing settings. The separate learned transition scorer kept the existing 50 m filter and excluded two trips at fit time. Every fold and test trip remained in evaluation.
- The previous frozen decoder scored 0.9065 on student test. The matched Phase 2 comparison uses its own gap-aware untuned decoder. CV was used to select settings. The ten student-test trips are a development benchmark, and reserved evaluator trips were outside this run.

## Timing and definitions

The tuned decoder's median student-test compute time was 99.6 ms per trip, including classifier scoring, transition setup from prepared graph features and dynamic programming. It excludes file I/O, candidate generation and shortest-path preprocessing.

Macro F1 averages arc classes within each trip, then averages trips. Top-1 is observation-weighted. Candidate log loss is binary and uses scorer probabilities before decoding. Direct continuity counts consecutive same-arc or forward-adjacent picks. Sequence similarity compares ordered sequences after collapsing consecutive duplicate arcs.

The tuned model's median raw-trip compute time, including candidate generation and graph preprocessing, was **314.4 ms**. This excludes loading the trip files.

# PathSnap pre-lunch demo

## What to say

> We are matching noisy GPS observations to the road segments a vehicle most likely used. The student archive has 80 training trips and 10 student-test trips. Each trip has GPS coordinates and timestamps in `.track`, road-node coordinates in `.nodes`, directed road segment endpoints in `.arcs`, and an ordered reference route in `.route`.
>
> For each GPS point we shortlisted 64 arcs by midpoint distance and kept the eight closest by point-to-segment distance. The models scored those eight options using features such as distance, heading agreement, movement between GPS readings, and road-segment length. The reference route gives a sequence of arcs, so we aligned it to the GPS readings in route order to obtain point-level labels.
>
> We compared five models using the same five whole-trip folds, then refit each model using the 80 training trips and scored all 10 student-test trips. Each fit used every second GPS observation for speed. Evaluation scored every observation. Gradient Boosting led every fold and scored 0.675 Macro F1 in cross-validation, then 0.690 Macro F1 and 83.74% Top-1 on student test. The nearest-road heuristic scored 53.86% Top-1.
>
> We randomly drew trip 00000076 for the map. On this trip boosting matched the aligned reference for 1,230 of 1,486 GPS observations, or 82.77%. Nearest segment matched 611, or 41.12%. The two markers show a correction and a remaining error.

## Click through the map

1. Open **PathSnap_Folium_Demo.html** in Edge or Chrome. The map and its road backdrop work offline.
2. Show the full trip. Blue dots are raw GPS, grey lines are nearby road geometry, dashed magenta is the supplied reference route, and teal is the set of arcs selected by boosting.
3. Turn on **Nearest-segment baseline** in the layer menu. If the lines overlap, turn off **Gradient Boosting predictions** briefly to isolate the baseline. Turn predictions back on before opening the examples.
4. Click **Show correction** and compare the reference, boosting and baseline arc IDs in the popup. Click **Show error** and point out that the aligned reference was among the candidates and boosting still chose another arc. Opposite directions can share the same line on the map, so compare arc IDs and the from-node → to-node direction.
5. Click **Full trip** to reset. Show the comparison chart after the map and describe the aggregate scores.

## Answers to likely questions

- **Dataset size.** 208,417 training GPS observations and 19,160 student-test observations. The demo trip has 26,635 road nodes, 56,027 directed arcs and 374 reference-route entries. The 10 reserved evaluator trips were outside the supplied ZIP.
- **Cleaning.** We checked file structure, IDs and timestamps, projected coordinates into metres, and derived features. We kept the raw GPS readings intact, including gaps and outliers, and did not interpolate or smooth them. We retained two training trips that together accounted for about 81% of training candidate misses.
- **Starting model settings.** Logistic Regression used C=1. The shallow tree used depth 5. Random Forest used 32 trees and depth 12. Histogram Gradient Boosting used 75 iterations, 31 maximum leaves per tree and a 0.10 learning rate. The shallow MLP used 32 hidden units and a 12-epoch limit.
- **Scores.** Top-1 is the share of GPS observations whose highest-scoring arc matches the aligned reference. Macro F1 weights arc classes equally within each trip and then averages trip scores. Candidate misses count as Top-1 errors. The miss rate was 1.12% across student test and 2.49% on this trip.
- **Scope.** The teal traces are geometry for per-point arc choices. Bayesian tuning, HMM/Viterbi reconstruction and the live two-laptop demo are final-delivery work.

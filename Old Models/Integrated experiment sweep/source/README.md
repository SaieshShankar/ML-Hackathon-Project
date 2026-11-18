# Integrated LightGBM experiment

This CPU run combines adjacent-candidate context, multiscale directional features, LightGBM, and topology-aware Viterbi. The original eight candidates and derived reference labels are reused. The raw dataset is distributed separately.

## Reproduce

Use Python 3.11 and install the pinned dependencies.

    python -m pip install -r integrated/requirements.txt

Extract the original five-model evidence archive into a folder named set1. Use benchmark.py from that archive to reconstruct its data.joblib cache if needed. Run the commands from the repository root. The optional report verifier uses the reference_comparisons directory in the integrated evidence bundle.

    python set1/benchmark.py prepare --dataset "Student folder.zip" --cache data.joblib --output set1-rebuild
    python integrated/prepare_integrated.py --dataset "Student folder.zip" --cache data.joblib --set1 set1 --output prepared
    python integrated/run.py --set1 set1 --prepared prepared --output integrated-rerun
    python integrated/verify.py --run integrated-rerun --set1 set1 --prepared prepared --archive reference_comparisons

The preparation command checks the ZIP digest against Set 1. The feature cache uses about 850 MB. The standalone preprocessor was checked against the original input arrays for a real student-test trip. Timing varies by CPU and thread availability.

## What was compared

- Three LightGBM scoring models share the original fitted-tree settings. The 33-feature model uses the original candidate features plus 23 features from neighboring candidate observations. The 51-feature models add the 18 multiscale heading features from the direction study. The original context and its rewritten version agree within 0.000002 on all ten student-test trips.
- The filtered models omit training trips whose median GPS-to-reference alignment distance exceeds 50 m. This removes trips 00000033 and 00000054 at fit time. The unfiltered scorer fits all trips. The logistic transition model is fit on retained trips for all variants to hold the transition learner fixed. All 80 training trips remain in validation.
- The decoder sweep compares the attached topology costs, emission weights, gap resets, directed travel distances, direction penalties, and learned logistic transition scores. Legacy mode assigns an exact reverse the direct-connection cost. Explicit reverse variants adjust those edges before decoding. Separate rules boost or penalise transitions based on GPS direction, so the experiment does not assume that all reversals are invalid.
- Selection uses mean observation-level Macro F1 over the original five equal-sized holdout trip folds. Trip IDs do not cross the fitting and validation sides of a fold. The settings are fixed before scoring the 10 student-test trips. The reserved evaluator trips are outside the experiment.
- Raw classifier and legacy decoder results for each refit are in student_test_comparison.csv inside this archive. The public-facing comparison shows one CV-chosen configuration per scoring model along with prior comparison points.

The trained LightGBM boosters are exported as native text models. The transition logistic scaler and weights are saved together. The code, logs, metrics, prediction arrays, and input provenance are included with the model artifacts.

# Patient Demo Data

This folder contains a small, labelled sample from UCI dataset 470, Parkinson's Disease Classification.

Important: UCI 470 provides pre-extracted feature rows, not the original patient audio recordings. The files here are therefore feature samples, not `.wav` files. They are useful for inspecting the trained detector and running tabular experiments, but they cannot be uploaded to the browser's audio `/screen` endpoint.

Contents:

- `uci470_feature_samples.csv`: 12 rows from four distinct speakers, three repetitions each. Speakers 0 and 1 are labelled `parkinson_patient`; speakers 11 and 14 are labelled `healthy_control`.
- `manifest.csv`: speaker, class, and repetition counts.

The model's normal application path remains:

```text
.wav recording -> feature extractor -> 107 model features -> detector_v2
```

Do not describe these public dataset rows as new patient recordings or as clinical validation. The dataset is a research benchmark and the model is a screening aid, not a diagnostic tool.

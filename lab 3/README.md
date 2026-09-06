# Lab 3: Healthcare Data Preprocessing and Analysis

Lab 3 studies healthcare data preparation across three notebooks and four data modalities: clinical text, chest X-ray images, tabular metadata, and physiological ECG signals.

## Notebooks

| Notebook | Modality | Main work | Data status |
|---|---|---|---|
| `lab_3_1.ipynb` | Clinical text | Missing-value handling, PHI replacement, text normalization, EDA, TF-IDF representation, specialty labels, and train/test split | Requires `mtsamples.csv` |
| `Lab_3_2.ipynb` | Chest X-ray imaging | Dataset inspection, class counts, corrupted-file checks, resizing, grayscale normalization, pixel EDA, filtering, contrast enhancement, edge detection, ResNet features, and augmentation | Requires the Kaggle chest X-ray dataset path used in the notebook |
| `lab_3_3.ipynb` | Tabular metadata and ECG signal | MIT-BIH header parsing, clinical plausibility checks, hypothesis testing, chi-square, ANOVA, feature extraction, filtering, signal windows, and augmentation | Runs on the included PhysioNet MIT-BIH files |

## Findings From `lab_3_1.ipynb`

- The workflow keeps `medical_specialty` and `transcription`, drops rows without transcription, and replaces missing specialties with `Unknown`.
- The PHI-cleaning function replaces email addresses, phone numbers, dates, and years with placeholders before lowercasing and removing punctuation.
- Exploratory analysis includes specialty frequencies, the top ten specialties, word-count distribution, and the most common words.
- TF-IDF is configured with English stop-word removal and a maximum vocabulary of 5,000 features.
- Specialty labels are encoded numerically, and the intended split is an 80/20 stratified train/test split.
- The notebook requires `TfidfVectorizer`, `LabelEncoder`, and `train_test_split` imports before the final modeling cells can run.
- No transcription CSV is included in this repository, so dataset size, specialty counts, and model performance cannot be reported from the checked-in files.

## Findings From `Lab_3_2.ipynb`

- The notebook expects the Kaggle Chest X-Ray Pneumonia dataset with `train`, `val`, and `test` folders containing `NORMAL` and `PNEUMONIA` classes.
- It checks folder contents, class counts, image dimensions and data types, corrupted images, file extensions, and pixel-intensity distributions.
- Images are converted to grayscale, resized to `224 x 224`, and normalized to `[0, 1]` for TensorFlow input.
- Gaussian and median filtering, histogram equalization, CLAHE, and Canny edge detection are explored as image-cleaning and feature-extraction operations.
- ResNet50 is used as an ImageNet feature extractor, and augmentation includes horizontal flips, rotation, zoom, and contrast changes.
- No image dataset is included locally, so image counts, pixel statistics, and extracted feature shapes depend on the external Kaggle dataset.
- For clinical use, the image workflow should additionally anonymize DICOM metadata when present and split data by patient rather than by image.

## Findings From `lab_3_3.ipynb`

### Tabular metadata

The notebook parses the 48 included MIT-BIH header files into recording-level metadata. The resulting table contains record ID, sampling frequency, sample count, duration, age, sex, annotation availability, and header notes.

- 48 records were found.
- Age was missing for 2 records, or 4.2%.
- Sex counts were 24 male, 22 female, and 2 unknown.
- Sampling frequency, sample count, and duration were constant across this metadata subset; they should not be treated as discriminative model features.
- Implausible ages are converted to missing values rather than clipped into valid-looking clinical values.
- Engineered features include an age-missing indicator, median imputation for age, log duration, standardization, and an SVD-based extraction step.
- The final model-ready tabular tensor has shape `(48, 5)` and dtype `float32`.

### Statistical tests

The tests were run on the available recording metadata, not on patient-level treatment outcomes:

- Welch two-sample test comparing age by sex: statistic `-0.222`, p-value `0.8258`.
- Chi-square test of sex versus age missingness: statistic `48.000`, 2 degrees of freedom, p-value approximately `3.775e-11`.
- One-way ANOVA comparing age across male and female groups: F `0.052`, p-value `0.8214`.

These results are descriptive only. The chi-square result mainly reflects that the two records with missing age are labelled `Unknown`; it should not be interpreted as a clinical association.

### ECG signal

- MIT-BIH record `100` was loaded at 360 Hz with 650,000 samples.
- The signal is NaN-safe, band-pass filtered from 0.5 to 40 Hz, baseline-corrected, robustly clipped, and standardized per window.
- The pipeline creates 180 non-overlapping 10-second windows of 3,600 samples each.
- Training-only amplitude jitter is used as augmentation; validation and test records must remain unaugmented.
- The raw and cleaned first ten seconds are plotted for visual quality control.

## Healthcare-Specific Lessons

Healthcare data needs stronger controls than generic datasets because:

- Missing values may be encoded as sentinel values or may carry clinical meaning.
- Implausible measurements can result from unit errors, device faults, or corrupted records.
- Class imbalance can make accuracy misleading; sensitivity, specificity, precision-recall, and calibration should also be reported.
- Text, timestamps, rare diagnoses, record IDs, and imaging metadata can expose PHI or enable re-identification.
- ECG and imaging data contain acquisition noise and artifacts that require modality-specific quality checks.
- Patient-level splitting is essential to prevent leakage across images, text notes, or signal windows from the same person.

## Reproducibility

Activate the project environment before running the notebooks:

```bash
source venv/bin/activate
python -m pip install numpy pandas matplotlib pillow scipy scikit-learn wfdb
```

Additional dependencies may be required for the original notebooks:

```bash
python -m pip install opencv-python seaborn tensorflow pydicom
```

The local signal dataset is under `dataset/physionet.org/files/mitdb/1.0.0`. The text and chest X-ray notebooks require their respective approved datasets to be downloaded separately, with source, license, version, and download date recorded before analysis.

# Heart Disease Prediction

A beginner machine learning project that explores whether a Logistic Regression model can predict a binary `target` label from a small set of patient measurements. The notebook demonstrates a basic supervised learning workflow: inspect a CSV, separate features and labels, create a train/test split, fit a classifier, and print predictions.

> **Educational project only.** This model and dataset are not validated for clinical use. Do not use them to diagnose, treat, or make health decisions.

## Project contents

| File | Purpose |
|---|---|
| `heart_disease.ipynb` | Loads and inspects the data, trains the classifier, and prints predictions for the test set. |
| `heart.csv` | The 16-row dataset read by the notebook. |

Keep both files in the same directory when running the notebook because it loads the CSV by the relative path `heart.csv`.

## Dataset overview

The CSV has 16 records and 11 columns: 10 input features and the `target` label. The notebook's inspection reports that every column has 16 non-null values. The target counts are:

| Target value | Number of records |
|---|---:|
| `0` | 5 |
| `1` | 11 |

The source and precise label meaning are not documented in the supplied files. Accordingly, this README refers to the label values as `0` and `1` without assigning a clinical interpretation to either one.

### Columns

| Column | Description in the project context |
|---|---|
| `age` | Age measurement |
| `sex` | Encoded sex value; encoding is not specified in the notebook |
| `cp` | Encoded chest-pain category; category mapping is not specified |
| `trestbps` | Resting blood pressure measurement |
| `chol` | Cholesterol measurement |
| `fbs` | Encoded fasting blood sugar indicator; encoding is not specified |
| `thalach` | Maximum heart rate achieved measurement |
| `exang` | Encoded exercise-induced angina indicator; encoding is not specified |
| `oldpeak` | ST depression measurement recorded in the data |
| `ca` | Encoded `ca` value; meaning and category mapping are not documented in the notebook |
| `target` | Binary label used as the prediction target (`0` or `1`) |

The CSV header spells the `ca` column as `ca ` with a trailing space. The notebook uses all columns other than `target` as features, so it includes that column as supplied. If the dataset is prepared for a different pipeline, the header should be checked and normalized consistently.

## Notebook workflow

1. **Load and inspect:** pandas reads `heart.csv`; the notebook displays the first five rows, DataFrame column types and non-null counts, and the target-value counts.
2. **Select inputs and label:** `target` is removed from the feature DataFrame (`x`) and assigned to the label series (`y`). The remaining ten columns are used as model inputs.
3. **Split the records:** scikit-learn's `train_test_split` uses `test_size=0.2` and `random_state=42`. For 16 records, this creates 12 training records and 4 test records.
4. **Train the classifier:** `LogisticRegression(max_iter=1000)` is fit on the training subset.
5. **Print predictions:** the model predicts labels for the test subset, and the notebook prints `pred[:10]`. Since there are only four test records, all four predictions are shown.

The notebook does not apply feature scaling, imputation, encoding transformations, or other preprocessing. Its displayed inspection reports no missing values in this particular CSV.

## Model and recorded output

The notebook trains one model: scikit-learn's Logistic Regression classifier. The recorded test-set prediction output is:

```text
[1 0 1 1]
```

This output contains predicted labels only. The notebook does not print the test-set true labels alongside the predictions, compute accuracy, precision, recall, F1 score, a confusion matrix, or compare against a baseline or another model. Therefore, no performance score or conclusion about predictive quality can be reported from the notebook as provided.

The test set contains only four records and the full dataset contains only 16. Any performance estimate from such a split would be highly sensitive to individual records. The results are best read as an illustration of model fitting and prediction, not as evidence that the model generalizes.

## Libraries and environment

The notebook uses:

- Python (the notebook metadata records Python 3.9.12)
- pandas for reading and inspecting the CSV
- scikit-learn for the train/test split and Logistic Regression model
- Jupyter Notebook or JupyterLab to run the `.ipynb` file

A minimal package install for the imports used in the executed cells is:

```bash
python -m pip install pandas scikit-learn jupyter
```

## Run it locally

1. Save or clone the project files into one directory.
2. Open a terminal in that directory and install the packages listed above.
3. Start Jupyter Notebook or JupyterLab and open `heart_disease.ipynb`.
4. Run the cells in order. Make sure `heart.csv` remains beside the notebook so the relative file path resolves.

## Current project scope

The supplied notebook is a compact demonstration of data loading, train/test splitting, classifier fitting, and prediction. It does not include a saved model, a prediction interface, plots, formal evaluation, or model comparison. The final cell contains an unexecuted import statement for `LogisticRegression` from `sklearn.tree`; the trained model earlier in the notebook correctly imports `LogisticRegression` from `sklearn.linear_model`.

## Possible next steps

For a future version of the project, consider documenting the dataset source and target encoding, reporting predictions beside their true labels, and adding evaluation metrics. With a larger and appropriately sourced dataset, a stratified evaluation strategy and a comparison to a simple baseline could make the results more informative. These are suggestions for further work; they are not results included in the current notebook.

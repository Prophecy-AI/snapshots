# Binary Classification with a Bank Dataset

**Competition:** playground-series-s5e8
**Category:** None
**Deadline:** 2025-08-31T23:59:00Z
**Reward:** None
**Evaluation Metric:** Roc Auc Score
**Teams:** 3365

---

## Evaluation

Submissions are evaluated using [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) between the predicted value and the observed target.


## Submission File
For each `id` in the test set, you must predict the probability of the binary target `y`. The file should contain a header and have the following format:

    id,y
    750000,0.5
    750001,0.5
    750002,0.5
    etc.

## Timeline

* **Start Date** - ${competition.DateEnabled}
* **Entry Deadline** - Same as the Final Submission Deadline
* **Team Merger Deadline** - Same as the Final Submission Deadline
* **Final Submission Deadline** -  ${competition.Deadline}

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## data-description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Bank Marketing Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full) dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

*   **train.csv** - the training dataset; `y` is the binary target
*   **test.csv** - the test dataset; your objective is to predict the probability `y` for each row
*   **sample_submission.csv** - a sample submission file in the correct format

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## abstract

**Welcome to the 2025 Kaggle Playground Series!** We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Your goal is to predict whether a client will subscribe to a bank term deposit.

## Prizes

* 1st Place - Choice of Kaggle merchandise
* 2nd Place - Choice of Kaggle merchandise
* 3rd Place - Choice of Kaggle merchandise

**Please note:** In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team. 
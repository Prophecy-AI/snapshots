# Predicting Optimal Fertilizers

**Competition:** playground-series-s5e6
**Category:** None
**Deadline:** 2025-06-30T23:59:00Z
**Reward:** None
**Evaluation Metric:** MAP@{K}
**Teams:** 2648

---

## Evaluation

Submissions are evaluated according to the Mean Average Precision @ 3 (MAP@3):

$$MAP@5 = \frac{1}{U} \sum\_{u=1}^{U} \sum\_{k=1}^{min(n,5)} P(k) \times rel(k)$$

where \\( U \\)  is the number of observations, \\( P(k) \\) is the precision at cutoff \\( k \\), \\( n \\) is the number predictions per observation, and \\( rel(k) \\) is an indicator function equaling 1 if the item at rank \\( k \\) is a relevant (correct) label, zero otherwise.

Once a correct label has been scored for *an observation*, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation. For example, if the correct label is `A` for an observation, the following predictions all score an average precision of `1.0`.

    [A, B, C, D, E]
    [A, A, A, A, A]
    [A, B, A, C, A]


## Submission File

For each `id` in the test set, you may predict up to 3 `Fertilizer Name` values, with the predictions space delimited.  The file should contain a header and have the following format:

    id,Fertilizer Name 
    750000,14-35-14 10-26-26 Urea
    750000,14-35-14 10-26-26 Urea 
    ...


## Timeline

* **Start Date** - ${competition.DateEnabled}
* **Entry Deadline** - Same as the Final Submission Deadline
* **Team Merger Deadline** - Same as the Final Submission Deadline
* **Final Submission Deadline** -  ${competition.Deadline}

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## data-description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Fertilizer prediction](https://www.kaggle.com/datasets/irakozekelly/fertilizer-prediction) dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

*   **train.csv** - the training dataset; `Fertilizer Name` is the categorical target
*   **test.csv** - the test dataset; your objective is to predict the  `Fertilizer Name` for each row, up to three value, space delimited.
*   **sample_submission.csv** - a sample submission file in the correct format.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## abstract

**Welcome to the 2025 Kaggle Playground Series!** We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Your objective is to select the best fertilizer for different weather, soil conditions and crops.

## Prizes

* 1st Place - Choice of Kaggle merchandise
* 2nd Place - Choice of Kaggle merchandise
* 3rd Place - Choice of Kaggle merchandise

**Please note:** In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team. 
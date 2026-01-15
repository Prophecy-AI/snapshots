# CSIRO - Image2Biomass Prediction

## Description

Farmers often walk into a paddock and ask one question: “Is there enough grass here for the herd?” It sounds simple, but the answer is anything but. Pasture biomass - the amount of feed available - shapes when animals can graze, when fields need a break, and how to keep pastures productive season after season. 

<img title=”Grass” src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Ff544570304d0394d8251a6bddf27c53a%2Fphoto-1533460004989-cef01064af7e.avif?generation=1757621128540049&alt=media" style="float: right; width: 300px">

Estimate incorrectly, and the land suffers; feed goes to waste, and animals struggle. Get it right and everyone wins: better animal welfare, more consistent production, and healthier soils.

Current methods make this assessment more challenging than it could be. The old-school “clip and weigh” method is accurate but slow and impossible at scale. Plate meters and capacitance meters can provide quicker readings, but are unreliable in variable conditions. Remote sensing enables broad-scale monitoring, but it still requires manual validation and can’t separate biomass by species.

This competition challenges you to bring greener solutions to the field: build a model that predicts pasture biomass from images, ground-truth measures, and publicly available datasets. You’ll work with a professionally annotated dataset covering Australian pastures across different seasons, regions, and species mixes, along with NDVI values to enhance your models.

If you succeed, you won’t just improve estimation methods. You’ll help farmers make smarter grazing choices, enable researchers to track pasture health more accurately, and drive the agriculture industry toward more sustainable and productive systems.

## Evaluation

## Scoring

The model performance is evaluated using a **globally weighted coefficient of determination (R²)** computed over all (image, target) pairs together.  
Each row is weighted according to its target type using the following weights:

- `Dry_Green_g`: 0.1  
- `Dry_Dead_g`: 0.1  
- `Dry_Clover_g`: 0.1  
- `GDM_g`: 0.2  
- `Dry_Total_g`: 0.5  

This means that instead of calculating R² separately for each target and then averaging, a single weighted R² is computed using all rows combined, with the above per-row weights applied.

---

## R² Calculation

The weighted coefficient of determination \\(R^2_w\\) is calculated as:

$$
R^2_w = 1 - \frac{\sum_j w_j (y_j - \hat{y}_j)^2}{\sum_j w_j (y_j - \bar{y}_w)^2}
$$

where \\(\bar{y}_w = \frac{\sum_j w_j y_j}{\sum_j w_j}\\).

**Residual Sum of Squares \\(SS_{\text{res}}\\)**  
Measures the total error of the model’s predictions:
$$
SS_{\text{res}} = \sum_{j} w_j (y_j - \hat{y}_j)^2
$$

**Total Sum of Squares \\(SS_{\text{tot}}\\)**  
Measures the total weighted variance in the data:
$$
SS_{\text{tot}} = \sum_{j} w_j (y_j - \bar{y}_w)^2
$$

**Terms**

- \\(y_j\\): ground-truth value for data point \\(j\\)  
- \\(\hat{y}_j\\): model prediction for data point \\(j\\)  
- \\(w_j\\): per-row weight based on target type  
- \\(\bar{y}_w\\): global weighted mean of all ground-truth values  

---


## Submission File

Submit a CSV in **long format** with exactly two columns:

- `sample_id` : ID constructed from image ID and target_name pair.
- `target`: Your predicted biomass value (grams) for that sample_id (float).

The valid `target` names are:
`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`.

Your file **must contain one row per (image, target) pair**, i.e., 5 rows for each image in the test set.

**Header and example:**

```csv
sample_id,target
ID1001187975__Dry_Green_g,0.0
ID1001187975__Dry_Dead_g,0.0
ID1001187975__Dry_Clover_g,0.0
ID1001187975__GDM_g,0.0
ID1001187975__Dry_Total_g,0.0
ID1001187976__Dry_Green_g,0.0
ID1001187976__Dry_Dead_g,0.0
ID1001187976__Dry_Clover_g,0.0
ID1001187976__GDM_g,0.0
ID1001187976__Dry_Total_g,0.0
```



## Timeline

* **October 28, 2025** - Start Date. 
* **January 21, 2026** - Entry Deadline. You must accept the competition rules before this date in order to compete. 
* **January 21, 2026** - Team Merger Deadline. This is the last day participants may join or merge teams. 
* **January 28, 2026** - Final Submission Deadline. 

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## data-description

# Competition Overview

In this competition, your task is to use pasture images to predict five key biomass components critical for grazing and feed management:

* **Dry green vegetation** (excluding clover)
* **Dry dead material**
* **Dry clover biomass**
* **Green dry matter (GDM)**
* **Total dry biomass**

Accurately predicting these quantities will help farmers and researchers monitor pasture growth, optimize feed availability, and improve the sustainability of livestock systems.

---

## Files

**test.csv**

* `sample_id` — Unique identifier for each prediction row (one row per image–target pair).
* `image_path` — Relative path to the image (e.g., `test/ID1001187975.jpg`).
* `target_name` — Name of the biomass component to predict for this row. One of: `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`.

The test set contains over 800 images.

**train/**

* Directory containing training images (JPEG), referenced by `image_path`.

**test/**

* Directory reserved for test images (hidden at scoring time); paths in `test.csv` point here.

**train.csv**

* `sample_id` — Unique identifier for each training *sample* (image).
* `image_path` — Relative path to the training image (e.g., `images/ID1098771283.jpg`).
* `Sampling_Date` — Date of sample collection.
* `State` — Australian state where sample was collected.
* `Species` — Pasture species present, ordered by biomass (underscore-separated).
* `Pre_GSHH_NDVI` — Normalized Difference Vegetation Index (GreenSeeker) reading.
* `Height_Ave_cm` — Average pasture height measured by falling plate (cm).
* `target_name` — Biomass component name for this row (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, or `Dry_Total_g`).
* `target` — Ground-truth biomass value (grams) corresponding to `target_name` for this image.


**sample\_submission.csv**

* `sample_id` — Copy from `test.csv`; one row per requested (image, `target_name`) pair.
* `target` — Your predicted biomass value (grams) for that `sample_id`.

---



### What you must predict

For each `sample_id` in **`test.csv`**, output a single numeric **`target`** value in **`sample_submission.csv`**. Each row corresponds to one `(image_path, target_name)` pair; you must provide the predicted biomass (grams) for that component. The actual test images are made available to your notebook at scoring time.

---
### Citation
 

Please cite this paper if you are using this dataset for research purposes.

 

```
@misc{liao2025estimatingpasturebiomasstopview,

      title={Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},

      author={Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie and Megan Walker and Joshua Whelan},

      year={2025},

      eprint={2510.22916},

      archivePrefix={arXiv},

      primaryClass={cs.CV},

      url={https://arxiv.org/abs/2510.22916},

}
```

## Code Requirements

<img src="https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png" style="float: right; width: 70px; padding: 10px">

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met: 
- CPU Notebook <= 9 hours run-time 
- GPU Notebook <= 9 hours run-time 
- Internet access disabled 
- Freely & publicly available external data is allowed, including pre-trained models 
- Submission file must be named `submission.csv` 

Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## abstract

Build models that predict pasture biomass from images, ground-truth measurements, and publicly available datasets. Farmers will use these models to determine when and how to graze their livestock. 

## Prizes

* 1st Place - $50,000
* 2nd Place - $20,000
* 3rd Place - $5,000

## Acknowledgements

Particular thanks given to our partner the Meat & Livestock Australia (MLA) for the images provided on the competition page. MLA is the declared industry marketing body and the industry research body for the Australian red meat industry. MLA’s mission is to collaborate with stakeholders to invest in research, development and marketing initiatives that contribute to producer profitability, sustainability and global competitiveness. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2F438f46cd9c7374478ebcf7f0d3f51c42%2Fmla_logo_home.png?generation=1757621105060923&alt=media)

This work has also been supported by FrontierSI (previously known as the Cooperative Research Centre for Spatial Information) ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2F913672f1c597e11b9ebc8ae9dbea4c30%2FFrontierSI_Logo_Primary%20(1).png?generation=1759373971687561&alt=media)

## About CSIRO

<img title=”logo” src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2F40387e5c32a2006adab8e2ee6294521d%2Fthumbnail.png?generation=1761574912631743&alt=media" style="float: right; width: 250px">

The Commonwealth Scientific and Industrial Research Organization (CSIRO) is Australia’s national science agency that is responsible for scientific research and its commercial and industrial applications.

At CSIRO, we solve the greatest challenges through innovative science and technology to unlock a better future for everyone. We are thinkers, problem solvers, leaders. We blaze new trails of discovery. We aim to inspire the next generation.

Working with industry, government, universities and research organisations we turn big ideas into disruptive solutions. Turning science into solutions for food security and quality; clean energy and resources; health and wellbeing; resilient and valuable environments; innovative industries; and a secure Australia and region.

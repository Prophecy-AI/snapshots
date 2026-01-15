# iNaturalist 2019 at FGVC6

**Competition:** inaturalist-2019-fgvc6
**Category:** None
**Deadline:** 2019-06-10T23:59:00Z
**Reward:** None
**Evaluation Metric:** MeanBestErrorAtK
**Teams:** 213

---

## data-description

## File descriptions
*   **train_val2019.tar.gz** - Contains the training and validation images in a directory structure following {iconic category name}/{category name}/{image id}.jpg . 
*   **train2019.json** - Contains the training annotations.
*   **val2019.json** - Contains the validation annotations.
*   **test2019.tar.gz** - Contains a single directory of test images.
*   **test2019.json** - Contains test image information. 
*   **kaggle_sample_submission.csv** - A sample submission file in the correct format.

## Image Format
All images have been saved in the JPEG format and have been resized to have a maximum dimension of 800 pixels. 

## Annotation Format
We follow the annotation format of the [COCO dataset][2] and add additional fields. The annotations are stored in the [JSON format][3] and are organized as follows:

    {
      "info" : info,
      "images" : [image],
      "categories" : [category],
      "annotations" : [annotation],
      "licenses" : [license]
    }

    info{
      "year" : int,
      "version" : str,
      "description" : str,
      "contributor" : str,
      "url" : str,
      "date_created" : datetime,
    }

    image{
      "id" : int,
      "width" : int,
      "height" : int,
      "file_name" : str,
      "license" : int,
      "rights_holder" : str
    }

    category{
      "id" : int,
      "name" : str,
      "kingdom" : str,
      "phylum" : str,
      "class" : str,
      "order" : str,
      "family" : str,
      "genus" : str
    }

    annotation{
      "id" : int,
      "image_id" : int,
      "category_id" : int
    }

    license{
      "id" : int,
      "name" : str,
      "url" : str
    }


## Description

![](https://www.dropbox.com/s/kltgq0ahtb05v4x/iNat_2019_github.gif?dl=1)

As part of the FGVC6 workshop at CVPR 2019 we are conducting the iNat Challenge 2019 large scale species classification competition,  sponsored by Microsoft. It is estimated that the natural world contains several million species of plants and animals. Without expert knowledge, many of these species are extremely difficult to accurately classify due to their visual similarity. The goal of this competition is to push the state of the art in automatic image classification for real world data that features a large number of fine-grained categories.

Previous versions of the challenge have focused on classifying large numbers of species. This year features a smaller number of highly similar categories captured in a wide variety of situations, from all over the world. In total, the iNat Challenge 2019 dataset contains 1,010 species, with a combined training and validation set of 268,243 images that have been collected and verified by multiple users from iNaturalist.

Teams with top submissions, at the discretion of the workshop organizers, will be invited to present their work at the FGVC6 workshop.  Participants who make a submission that beats the sample submission can fill out this [form][2] to receive $150 in Google Cloud credits. 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_logo_%282012%29.svg/200px-Microsoft_logo_%282012%29.svg.png)
<p><br><i>Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.</i></p>
  [1]: http://inaturalist.org 
 [2]: https://www.kaggle.com/GCP-Credits-CVPR2019

## Evaluation

We use top-1 classification error as the metric for this competition. For each image, an algorithm will produce 1 label.  If the predicted label matches the ground truth label then the error for that image is 0, otherwise it is 1. The final score is the error averaged across all images. 

## Submission File
For each image&nbsp;in the test set, you must predict 1 category label. However, we encourage you to predict more categories labels (sorted by confidence) so that we can analyze top-3 and top-5 performances. The csv file should contain a header and have the following format:

    id,predicted  
    268243,71 108 339 341 560  
    268244,333 729 838 418 785  
    268245,690 347 891 655 755

The `id` column corresponds to the test image id. The `predicted` column corresponds to 1 category id. The first category id will be used to compute the metric. You should have one row for each test image.

## CVPR 2019

This competition is part of the Fine-Grained Visual Categorization [FGVC6][1] workshop at the Computer Vision and Pattern Recognition Conference [CVPR 2019][2]. A panel will review the top submissions for the competition based on the description  of the methods provided. From this, a subset may be selected to present their results at the workshop. Attending the workshop is not required to participate in the competition, however only teams that are attending the workshop will be considered to present their work.

There is no cash prize for this competition. Attendees presenting in person are responsible for all costs associated with travel, expenses, and fees to attend CVPR 2019.


  [1]: https://sites.google.com/view/fgvc6/home
  [2]: http://cvpr2019.thecvf.com/

## Timeline

* **June 3, 2019** - Entry deadline. You must accept the competition rules before this date in order to compete.
* **June 3, 2019** - Team Merger deadline. This is the last day participants may join or merge teams.
* **June 10, 2019** - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
# Santa 2025 - Christmas Tree Packing Challenge

**Competition:** santa-2025
**Category:** None
**Deadline:** 2026-01-30T23:59:00Z
**Reward:** None
**Evaluation Metric:** Santa 2025 Metric
**Teams:** 3302

---

## Description

<img title=”seasons greetings” src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F8939556%2Ff84b88f18d9ee1657b7229ad9fab9713%2FGemini_Generated_Image_kgcl4gkgcl4gkgcl.png?generation=1762216480960277&alt=media" style="float: right; height: 357px">

_Here comes a challenge, here comes a challenge,_<br>
_Right to your front door! _ <br>
_Santa has tree toys, tiny tree toys,_ <br>
_To mail from shore to shore._ <br>
_He needs the smallest box, indeed a square box,_<br>
_To fit them all inside,_<br>
_So he can mail these stocking stuffers_<br>
_On his big long Christmas ride!_ <br>

_Here comes the problem, here comes the problem._<br>
_We need the smallest size!_<br>
_For one to two hundred trees in shipments,_<br>
_We need your expert eyes._<br> 
_Can you find the best solution to help us pack_<br> 
_All the tiny trees inside?_<br>
_We must find the optimal packing to help Santa Claus_<br> 
_And win a prize!_<br>


**The Challenge**

In this re-defined optimization problem, help Santa fit Christmas tree toys into the smallest (2-dimension) parcel size possible so that he can efficiently mail these stocking stuffers around the globe. Santa needs the dimensions of the smallest possible square box that fits shipments of between 1-200 trees. 

**The Goal**

Find the optimal packing solution to help Santa this season and win Rudolph's attention by being one of the first to post your solution! 

Happy packing!  

## Evaluation

Submissions are evaluated on sum of the normalized area of the square bounding box for each puzzle. For each `n`-tree configuration, the side `s` of square box bounding the trees is squared and divided by the total number `n` of trees in the configuration. The final score is the sum of all configurations. Refer to the [metric notebook](https://www.kaggle.com/code/metric/santa-2025-metric) for exact implementation details.

$$ \text{score} = \sum_{n=1}^{N} \frac{s_{n}^2}{n}$$

## Submission File
For each `id` in the submission (representing a single tree in a `n`-tree configuration), you must report the tree position given by `x`, `y`, and the rotation given by `deg`. To avoid loss of precision when saving and reading the files, the values must be converted to a string and prepended with an `s` before submission. Submissions with any overlapping trees will throw an error. To avoid extreme leaderboard scores, location values must be constrained to \\(-100 \le x, y \le 100\\).

The file should contain a header and have the following format:

    id,x,y,deg
    001_0,s0.0,s0.0,s20.411299
    002_0,s0.0,s0.0,s20.411299
    002_1,s-0.541068,s0.259317,s51.66348
    etc.

## Timeline

* **November 17, 2025** - Start Date.

* **January 23, 2026** - Entry Deadline. You must accept the competition rules before this date in order to compete.

* **January 23, 2026** - Team Merger Deadline. This is the last day participants may join or merge teams.

* **January 30, 2026** - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## data-description

The objective of this challenge is to arrange Christmas tree toys into the smallest packing arrangement possible&mdash;as defined by the size of a square bounding box around the trees&mdash;for toy counts of 1-200.

## Files

**sample_submission.csv** - a sample submission in the correct format
 - `id` - a combination of the `n`-tree count for the puzzle and the individual tree index within the puzzle
 - `x`, `y` - the 2-d coordinates of the tree; this point is defined at the center of the top of the trunk
 - `deg` - the rotation angle of the tree


This [Getting Started Notebook](https://www.kaggle.com/code/inversion/santa-2025-getting-started) implements a basic greedy algorithm (which was used to construct the `sample_submission.csv`), demonstrates collision detection, and provides a visualization function.

**Note:** The [metric](https://www.kaggle.com/code/metric/santa-2025-metric) for this competition has been designed to reasonably maximize floating point precision during calculations, but cannot guarantee precision beyond what is displayed on the leaderboard.

## Prizes

- First Prize: $12,000
- Second Prize: $10,000
- Third Prize: $10,000
- Fourth Prize: $8,000

**Rudolph Prize** - $10,000: Awarded to the team holding 1st place on the leaderboard for the longest period of time between November 17, 2025 12:00 AM UTC and January 30, 2026 11:59 PM UTC. In the event the competition needs to be restarted, the Rudolph Prize dates shall be the new start and deadline of the competition.

As a condition to being awarded a Prize, a Prize winner must provide a detailed write-up on their solution in the competition forums within 14 days of the conclusion of the competition.


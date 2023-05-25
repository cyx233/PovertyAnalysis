# Final Project - Poverty Analysis
In this final project, we study classification in the context of the [Poverty dataset](https://wilds.stanford.edu/datasets/#povertymap), which is part of the [Wilds Project](https://wilds.stanford.edu/).

The goal of this project is to classify poor vs. wealthy regions in Africa based on satellite imagery. There are `~20,000` images covering `23` countries in Africa.

The satellite images are of the shape `224 X 224 X 8`. Each pixel corresponds to a `30m X 30m` area and each image corresponds to a `6.7km X 6.7km` square. To see some sample images, see [this notebook](<https://github.com/SateeshKumar21/PovertyAnalysis/tree/main/HW5/Pre-processing/2.browse images.ipynb>). 

This dataset comprises images from both urban and rural areas. In general, urban areas are significantly more wealthy than rural areas. See [this notebook](<https://github.com/SateeshKumar21/PovertyAnalysis/tree/main/HW5/Pre-processing/2.browse images.ipynb>) for details. To make the problem into a classification task, we define a threshold on the wealth that separates the poor from wealthy. As there is a large difference between rural and urban, we use a different threshold for each subset. Rural images with wealth less than -0.5 are labeled as poor and greater than -0.5 as wealthy. Similarly, we pick a threshold of 1.3 for urban images.


## Dataset 
You can find the image files at the following location:
1. DataHub:
   - `/home/username/public/cs255-sp22-a00-public/poverty`
3. Vocareum: 
   - `Poverty Analysis Spring 2023/resource/asnlib/publicdata/anon_images/`
   - `Poverty Analysis Spring 2023/resource/asnlib/publicdata/train.csv`
   - `Poverty Analysis Spring 2023/resource/asnlib/publicdata/random_test_reduct.csv`
   - `Poverty Analysis Spring 2023/resource/asnlib/publicdata/country_test_reduct.csv`

All files (train and test) are stored inside this folder in npz format. We divided this dataset into one train and 2 test sets. We separated out ~25% of the data to build a countries test set (`Country_test_reduct.csv`) s.t. the test countries that are not present in the training set. In the random test set, we separated 25% of the instances at random from the remaining 75% of data to generate a random test set (`Random_test_reduct.csv`).

So, there are 3 csv files:
1. `train.csv`: Ground truth for the training dataset. Use the column `label` to train your models.
2. `Country_test_reduct.csv`: Country test set. You have all the same columns as `train.csv` except for `label` and `wealthpooled`.
3. `Random_test_reduct.csv`: Random test set. You have all the same columns as `train.csv` except for `label` and `wealthpooled`.

Note, random test set is an easier one and follows the same distribution as the train set. Country test set is harder as it consists of countries that you will not encounter in the train set. 


## Starting Point 
We provide a working solution to the HW, on which you are expected to improve. This starting point is provided in [this GitHub repository](https://github.com/SateeshKumar21/PovertyAnalysis).

The starting point includes a solution based on KDTree pre-processing + XGBoost. You can run the solution on Datahub/Vocareum, but it would run on most laptops/work-stations as well.

## Performance Evaluations
We will evaluate you on two different test sets using two different metrics.

The first metric is **asymmetric loss**. We assign weights to your predictions. That is, for every wrong prediction (poor classified as wealthy or vice versa), you get -2. For every correct prediction, you get +1. You can also give a prediction as “I don’t know” for which you get 0. We sum all these points and divide by the total number of test images.

The second metric is **symmetric loss**. This is similar to the asymmetric loss but with a wrong prediction penalty of -1. 

We have 2 competitions:
1. Test on random test set using symmetric loss.
4. Test on countries test set using asymmetric loss.

Each competition has a separate leaderboard.


## Compute Resources
#### DataHub
The training of the models can be done on Datahub. There is a separate container allocated for you with GPU support and PyTorch CUDA integrated.

Container Name: `ucsdets/scipy-ml-notebook (8 CPU, 16G RAM, 1 GPU)`

#### Vocareum
TBD

## How to Use the Autograder

Go to https://www.gradescope.com/ and enroll in CSE 255 (please sign in with your student email and use enrollment code: RZ4BJ5). DSC 232R students should also enroll in the same gradescope assignment. 

Go to `HW Assignment 5` and submit your files. You will be re-routed to a page where the autograder results will appear, as soon as they finish processing.

There are four files to submit:
1. `results.csv` — your predictions on the random test set 
2. `results_country.csv` — your predictions on the country test set 
3. `code.zip` or `code.tgz` — your code in a zip file, max size of zip file is 10MB.
4. `explanation.md` - your explanation for your implementation

Each csv file should have the following columns:

- `filename` — e.g. `image13724.npz`
- `urban` — `1` when urban, `0` when not urban
- `pred_with_abstention`  — predictions of `-1`, `1`, and `0` when I don’t know
- `pred_wo_abstention` - predictions of `-1`, `1` 

## Startup Code
Github repo: [Poverty Analysis](https://github.com/SateeshKumar21/PovertyAnalysis)

The repository contains a baseline which uses XGBoost. 

## Teams

Teams can consist of 1-4 members. Teams have to be chosen by **29th May 2023** and cannot be changed. We will open a dummy gradescope assignment for team selection.
By default, each student is in their own team. The grade of all members of a team is the grade given by the team.

## Grades 

The final scores will be curved and the grade will be scaled according to the formula `score=50+ (percentile/2)`. The average of the two scores will be used.

## Evaluation 
The feedback on each submission consists of two average scores - corresponding to the following combinations:
   * Random test / No abstension
   * Country Test / Abstension
   
These will also appear on a class leaderboard. The name under which it appears is your team name.  Please note that the asymmetric loss can be a value between -2 and 1, and appears in the leaderboard that way. In your evaluation test cases, this value is mapped to a number between 0 and 10 so that you don't get negative points.

## Number of Submissions Per Day
Each group can make at most one submission per group member per 24 hour period.


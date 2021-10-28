# Documentation Example

Some introductory sentence(s). Data set and task are relatively fixed, so 
probably you don't have much to say about them (unless you modifed them).
If you haven't changed the application much, there's also not much to say about
that.
The following structure thus only covers preprocessing, feature extraction,
dimensionality reduction, classification, and evaluation.

# Major pipeline changes
Due to major errors in executing the pipeline on my Linux Machine, I had to
change the folder name of `code` into `src`. Therefore, I had to change some
bash commands accordingly.

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Preprocessing

I'm following the "Design Decisions - Results - Interpretation" structure here,
but you can also just use one subheading per preprocessing step to organize
things (depending on what you do, that may be better structured).

### Design Decisions

Which kind of preprocessing steps did you implement? Why are they necessary
and/or useful down the road?

We have grouped the different media columns (namely `photos` and `video`) into one `media` column. This works because 
Twitter does not allow you to have both a video and a photo. This column can have up to 3 labels: `None`, `Video` and 
`Photo`, but it can be focused just on `Photos` as well. The argument parser accepts `photo`, `video`, `both` and `none`
However, now it is a categorical feature instead of being only a string before. The dataset column `Video` has a strange
property. It is '1', not just if the tweet contains a video, but a photo as well. We had to solve this issue. 
### Results

Maybe show a short example what your preprocessing does.

#### Daytime Example

adds new column based on the time the tweet was sent
The end hours of every daytime can be changed within the `-d` or `--daytime` argument.
By default, this is:
0  - 6  -> night
6  - 12 -> morning 
12 - 18 -> afternoon
18 - 0  -> evening
295806    18:29:04  ->  Evening
295807    07:46:31  ->  Morning
295808    03:24:09  ->  Night
295809    00:29:44  ->  Night
295810    19:45:15  ->  Evening

### Interpretation

Probably, no real interpretation possible, so feel free to leave this section out.

## Feature Extraction

Again, either structure among decision-result-interpretation or based on feature,
up to you.

### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

### Interpretation

Can we already guess which features may be more useful than others?

## Dimensionality Reduction

If you didn't use any because you have only few features, just state that here.
In that case, you can nevertheless apply some dimensionality reduction in order
to analyze how helpful the individual features are during classification

### Design Decisions

Which dimensionality reduction technique(s) did you pick and why?

### Results

Which features were selected / created? Do you have any scores to report?

### Interpretation

Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

## Classification

### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
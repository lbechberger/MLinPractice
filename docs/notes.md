

Can I link to the image ? 

- o [hard coded domainrepo+ branch + file/page](https://github.com/TobiObeck/MLinPractice/blob/documentation-2/docs/Documentation.mdtweets-language)
- o [hard coded file/page](Documentation.mdtweets-language)
- o [hard coded file/page](./Documentation.mdtweets-language)
- [full path?](/docs/Documentation.mdtweets-language)
- [opens docs folder](./tweets-language)
- ++ [relative same page](tweets-language)
- + <a href="tweets-language">relative same page but html</a>

![asd](imgs/after_sentiment_2021-11-03_231550.png " ")


# visualization.py

![asd](imgs/distribution_of_tweets_per_language.png " ")



- Number of tweets: 295811
- Label distribution:
- False    0.908185
- True     0.091815


| en | 283240 |
|----|--------|
| es | 3492   |
| fr | 3287   |
| de | 811    |
| it | 748    |

- 283240 english tweets
- 12571 non english tweets
- 4.438 percent non-english
- 4.4382855528880105


Here's a simple footnote,[^1] and here's a longer one.[^bignote]. And another one [^2] here


<p align="center">
    <img id="distribution-of-tweets-per-language" src="./imgs/distribution_of_tweets_per_language.png" alt="">

Fig. 2: The majority of tweet records are labelled as english. The amount of non-english tweets is too small to be usefull for machine learning.
</p>


---

## References

[^1]: This is the first footnote.

[^1]: This is the first footnote.

[^bignote]: Here's one with multiple paragraphs and code.

    Indent paragraphs to include them in the footnote.

    `{ my code }`

    Add as many paragraphs as you like.

[^2]: This is the first footnote.

| index | param_criterion | param_min_samples_split | param_n_estimators | mean_test_cohen_kappa | rank_test_cohen_kappa | mean_test_rec | rank_test_rec | mean_test_prec | rank_test_prec | rank_sum |
|-------|-----------------|-------------------------|--------------------|-----------------------|-----------------------|---------------|---------------|----------------|----------------|----------|
| 19    | entropy         | 5                       | 101                | 0.906                 | 2                     | 0.144         | 35            | 0.509          | 2              | 39       |
| 20    | entropy         | 5                       | 121                | 0.906                 | 3                     | 0.143         | 36            | 0.508          | 3              | 42       |
| 41    | gini            | 5                       | 121                | 0.907                 | 1                     | 0.140         | 42            | 0.514          | 1              | 44       |
| 32    | gini            | 4                       | 81                 | 0.905                 | 11                    | 0.153         | 24            | 0.491          | 10             | 45       |
| ...   | ...             | ...                     | ...                | ...                   | ...                   | ...           | ...           | ...            | ...            | ...      |
| 14    | entropy         | 5                       | 1                  | 0.867                 | 38                    | 0.217         | 5             | 0.258          | 40             | 83       |
| 29    | gini            | 4                       | 21                 | 0.903                 | 31                    | 0.158         | 20            | 0.462          | 32             | 83       |
| 0     | entropy         | 3                       | 1                  | 0.866                 | 41                    | 0.225         | 3             | 0.257          | 41             | 85       |
| 21    | gini            | 3                       | 1                  | 0.864                 | 42                    | 0.225         | 2             | 0.250          | 42             | 86       |
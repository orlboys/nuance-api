Since CSV files cannot have comments, I've put the information about each of the datasets into this txt file.

## 'political_bias.csv'

### General Info:

- Sourced from https://huggingface.co/datasets/cajcodes/political-bias
- Structure is text, label.
  - The 'text' is a short opinion statement about a poltical topic.
  - The 'label' is a number in range 0 to 4 which indicates the bias of the opinion.
    - 0 is far right, 4 is far left.

### Data Overview:

#### Text Length Distribution

| Statistic       | Value |
| --------------- | ----- |
| Count           | 657   |
| Mean            | 10.60 |
| Std Dev         | 1.54  |
| Min             | 7     |
| 25th Percentile | 10    |
| Median (50th)   | 10    |
| 75th Percentile | 11    |
| Max             | 19    |

#### Class Distribution

| Label | Count | Proportion (%) |
| ----- | ----- | -------------- |
| 0     | 55    | 8.37           |
| 1     | 206   | 31.35          |
| 2     | 201   | 30.59          |
| 3     | 119   | 18.11          |
| 4     | 76    | 11.57          |

### Analysis Outcomes:

- SIGNIFICANTLY MORE WEIGHT must be given to labels 3, 4, and 0.
  - The class distribution shows moderate imbalance. Classes 1 and 2 are the most frequent, while Class 0 is underrepresented.
  - Luckily, this is ok because we have weight balancing!
- The maximum length of a sentence is 19 words, and the mean is 10 - it can be done by BERT but barely.

## 'allsides_data_unstructured.csv'

### General Info:

- Sourced from https://huggingface.co/datasets/Faith1712/Allsides_political_bias_proper
- Structure is text, label
  - The 'text' is an opinion statement
  - The 'label' is the bias classification of the bias
    - 0 is left, 2 is right, 1 is centre.
  - Based off the AllSides political bias dataset

## ./jsons

### General Info

- New approach - lets use JSONS!!
- 37554 different entries, with more data than before on sources and texts.
- more structured data reading, since everything is now in jsonified form (is it obvious I love jsons yet)
- Sourced from https://github.com/ramybaly/Article-Bias-Prediction
- Structure is "topic", "source", "bias", "url", "title", "date", "authors", "content", "content_original", "source_url", "bias_text", "bias"
  - For now, just using "content" and "bias"
  - Bias is 0 - 2 scale (left to right).
  - Crawled from https://www.allsides.com/unbiased-balanced-news

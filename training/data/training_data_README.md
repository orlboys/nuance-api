Since CSV files cannot have comments, I've put the information about each of the datasets into this txt file.

## 'political_bias.csv'

### General Info:
- Sourced from https://huggingface.co/datasets/cajcodes/political-bias
- Structure is text, label.
    - The 'text' is a short opinion statement about a poltical topic.
    - The 'label' is a number in range 0 to 4 which indicates the bias of the opinion.
        - 0 is far right, 4 is far left.

### Data analysis:

#### Text Length Distribution:
count    657.000000
mean      10.604262
std        1.544615
min        7.000000
25%       10.000000
50%       10.000000
75%       11.000000
max       19.000000

#### Class Proportions:
label
0    0.083714
1    0.313546
2    0.305936
3    0.181126
4    0.115677

#### Label Counts:
label
1    206
2    201
3    119
4     76
0     55

### Analysis Outcomes:

- SIGNIFICANTLY MORE WEIGHT must be given to labels 3, 4, and 0.
- The maximum length of a sentence is 19 words, and the mean is 10 - it can be done by BERT but barely.
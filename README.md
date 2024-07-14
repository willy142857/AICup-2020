
# 醫病訊息決策與對話語料分析競賽 - 秋季賽：醫病資料去識別化 - Team name : Lobster

## Final Score
Achieved **0.811 F1-score** and received honorable mention [**(6/174)**](https://www.aicup.tw/_files/ugd/7fbdbf_4ed126ff1bb34c19b39f3d476361210d.pdf) in the competition.

## Getting Started

### Prerequisites

We test on the Ubuntu 16.04 and a Nvidia RTX 2080 TI card.

* Ubuntu 16.04
* CUDA 10.1
* Python 3.8

### Installing

for `bert-based/`, you shoud install the following package:

* pytorch 1.7
* scikit-learn 0.23.2
* pandas 1.1.3
* tqdm 4.1.1
* transformers 3.5.1
* datasets 1.1.3
* seqeval 1.2.2

And for `bert-crf`, you shoud install:

* pytorch 1.7
* transformers 2.8

## How to run

### For pretrained model

1. Download model.zip from <https://drive.google.com/file/d/1uId99To_xoEa0-KZ7PahYsz09ysziqbm/view?usp=sharing>
2. run `example.sh`

3. run

   ```sh
   cd bert-based && scripts/predict.sh
   ```

4. run

   ```sh
   cd bert-crf/scripts && ./run_crf_ner_pred.sh 
   ```

5. run `bert-based/generate_pred.ipynb`
6. `bert-based/outputs-tsv/result.tsv` is the prediction.

### train by yourself

```sh
./bert-based.sh # train the bert-based model and get the prediction
./bert-crf.sh # train the bert-crf model and get the prediction
```

run `bert-based/generate_pred.ipynb` and get the final prediction.

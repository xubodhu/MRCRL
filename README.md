#Improving Sentence-Level Relation Classification via Machine Reading Comprehension and Reinforcement Learning

##Request

* tensorflow>=1.13
* tqdm>=4.55.1
* transformers>=3.5.1
* torch>=1.3.0
* numpy>=1.19.4
* tokenizers>=0.9.3

##Data
The original train data file of NYT is `train.txt`. In this paper, we divided 10% of the data as the validation set `valid_0.1.txt`, and the remaining 90% of the data as the training set `train_0.9.txt`.
`tp_tenp.txt` and `tp_data.txt` represent the test sets `NYT-T1` and `NYT-T2`, respectively. `relation2question.txt` is the question template.
In addition, you need to download the pre-trained `deepset/roberta-base-squad2` model, the [Baidu Yun](https://pan.baidu.com/s/16CMxDt2d6DuLN2_MSb4GoQ) password is:ty4h.
Note that the `roberta-base-squad2` folder is placed in the `deepset` directory and other files are placed directly in the root directory.
* train.txt
* train_0.9.txt
* valid_0.1.txt
* relation2question.txt
* tp_data.txt
* tp_tenp.txt
* vec.txt

##Initial
In this experiment, you need to run two files to get the initial embedding in the training phase and the initial embedding in the inference phase before starting the training:
```python
python get_train_embedding.py
```
```python
python get_test_embedding.py
```

##Training
```python
python RL_MRC.py
```
##Inference
```python
python inference.py
```
For more implementation details, please read the source code.


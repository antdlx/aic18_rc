## Intruduction
This project is my code for [AiChallenger2018 Opinion Questions Machine Reading Comprehension](https://challenger.ai/competition/oqmrc2018). This project mainly has 2 models which implemented in Tensorflow.
* Model-1 is based on QANet, but I rewrite it in some details.
* Model-2 is based on capsuleNet which mainly from [freefuiiismyname's project](https://github.com/freefuiiismyname/capsule-mrc)

## Dependencies
* Python 3.6
* Tensorflow 1.9.0
* tqdm
* gensim

## Data Sample
{
“query_id”:1,
“query”:“维生c可以长期吃吗”,
“url”: “xxx”,
“passage”: “每天吃的维生素的量没有超过推荐量的话是没有太大问题的。”,
“alternatives”:”可以|不可以|无法确定”,
“answer”:“可以”
}

## Performance
I train each model on one GTX1080ti for 30 epochs, and report the best Performance on dev set. We finally run Model-1 on testa, the accuracy is 73.2.

Model | Accuracy
---|---
Model-1 | 73.66
Model-2 | 73.85
ensembled | 76.62

## Project Structure

* capsuleNet： Model-2's codes
* data: Model-2's data
* QANet: Model-1's codes and data
* start.sh: example for usage
* vote_ser_new_word.py: vote ensemble file

## Details
you can see more details in /QANet/README.md and /capsuleNet/README.md

## Reference
some codes are borrowed from :

[NLPLearn/QANet](https://github.com/NLPLearn/QANet)

[freefuiiismyname/capsule-mrc](https://github.com/freefuiiismyname/capsule-mrc)

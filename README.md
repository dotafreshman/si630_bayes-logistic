# si630_bayes-logistic

## Naive Bayes

### tokenize, ngram_tokenize	

how to split sentence into words

* `Tokenize`	recommended orginal tokenize

* `ngram_tokenize`   use n-gram. You can change n by change variable `ngram`

  There are three times in the file tokenize are used. If you want to change method, you should change them all  

### alpha_test

plot a line, with smoothing factor alpha as x, accuracy as y to decide which alpha is best. User can determine range of alpha by `st` `ed` `intv` 

### variable `op`

* `test`  the model is trained on `train.txt`, and tested on `dev.txt`

* `output`  the model is tested on `test.txt` , and output label on csv file

## logistic regression 

also has `op` variable can be set to `test` or `out` , you can adjust learning rate and steps of training 

  	
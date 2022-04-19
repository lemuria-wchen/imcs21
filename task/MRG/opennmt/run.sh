## seq2seq (lstm)

# build the vocab
onmt_build_vocab -config lstm.yaml -n_sample 10000

# train
onmt_train -config lstm.yaml

# inference
onmt_translate -model saved/lstm_step_20000.pt -src data/src-test.txt -output data/pred_lstm.txt -gpu 0


## pointer generator

# build the vocab
onmt_build_vocab -config pg.yaml -n_sample 10000

# train
onmt_train -config pg.yaml

# inference
onmt_translate -model saved/pg_step_20000.pt -src data/src-test.txt -output data/pred_pg.txt -gpu 0


## transformer
# build the vocab
onmt_build_vocab -config transformer.yaml -n_sample 10000

# train
onmt_train -config transformer.yaml

# inference
onmt_translate -model saved/tf_step_20000.pt -src data/src-test.txt -output data/pred_tf.txt -gpu 0

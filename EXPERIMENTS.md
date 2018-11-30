# Experiments

11/23/18:

- Ran:
  - NB: (by mistake,) this is not the autoregressive model! to train the autoregressive model, we need to specify the `-ar_after_decay` flag.

To train the model:

```
time python chsmm.py
-data data/e2e_aligned \
-emb_size 300 \
-hid_size 300 \
-layers 1 \
-K 55 \
-L 4 \
-log_interval 200 \
-thresh 9 \
-emb_drop \
-bsz 8 \
-max_seqlen 55 \
-lr 0.5 \
-sep_attn \
-max_pool \
-unif_lenps \
-one_rnn \
-Kmul 5 \
-mlpinp \
-onmt_decay \
-cuda \
-seed 1818 \
-save models/chsmm-e2e-300-55-5.pt\  # K=55, Kmul=5?
```

11/25/18:

- Ran:
  - NB: (by mistake,) this is not the autoregressive model! to train the autoregressive model, we need to specify the `-ar_after_decay` flag.

To do the template extraction (Viterbi segmentation):

```
time python chsmm.py \
-data data/e2e_aligned \
-emb_size 300 \
-hid_size 300 \
-layers 1 \
-K 55 \
-L 4 \
-log_interval 200 \
-thresh 9 \
-emb_drop \
-bsz 8 \
-max_seqlen 55 \
-lr 0.5  \
-sep_attn \
-max_pool \
-unif_lenps \
-one_rnn \
-Kmul 5 \
-mlpinp \
-onmt_decay \
-cuda \
-load models/chsmm-e2e-300-55-5.pt.1 \  # K=55, Kmul=5?
-label_train | tee segs/seg-e2e-300-55-5.txt
```

11/26/18:

- Ran:
  - NB: (by mistake,) this is not the autoregressive model! to train the autoregressive model, we need to specify the `-ar_after_decay` flag.

To do generation:

11/28/18:

- Ran:
  - ensure that we have the same `-K` used when model was trained
  - ensure that we have the same `-Kmul` used when model was trained
  - this is for the (deduped) validation set

```
time python chsmm.py \
-data data/e2e_aligned/ \
-emb_size 300 \
-hid_size 300 \
-layers 1 \
-K 55 \
-dropout 0.3 \
-L 4 \
-log_interval 100 \
-thresh 9 \
-lr 0.5 \
-sep_attn \
-unif_lenps \
-emb_drop \
-mlpinp \
-onmt_decay \
-one_rnn \
-Kmul 5 \
-max_pool \
-gen_from_fi data/e2e_aligned/src_uniq_valid.txt \
-load models/chsmm-e2e-300-55-5.pt.1 \
-tagged_fi segs/seg-e2e-300-55-5.txt \
-beamsz 5 \
-ntemplates 100 \
-gen_wts '1,1' \
-cuda \
-min_gen_tokes 0 \
-seed 1 > gens/20181129-gen-e2e-300-55-5-src_uniq_valid.txt
```

- Ran:
  - ensure that we have the same `-K` used when model was trained
  - ensure that we have the same `-Kmul` used when model was trained
  - this is for the (deduped) test set

```
time python chsmm.py \
-data data/e2e_aligned/ \
-emb_size 300 \
-hid_size 300 \
-layers 1 \
-K 55 \
-dropout 0.3 \
-L 4 \
-log_interval 100 \
-thresh 9 \
-lr 0.5 \
-sep_attn \
-unif_lenps \
-emb_drop \
-mlpinp \
-onmt_decay \
-one_rnn \
-Kmul 5 \
-max_pool \
-gen_from_fi data/e2e_aligned/src_test.txt \
-load models/chsmm-e2e-300-55-5.pt.1 \
-tagged_fi segs/seg-e2e-300-55-5.txt \
-beamsz 5 \
-ntemplates 100 \
-gen_wts '1,1' \
-cuda \
-seed 1 \
-min_gen_tokes 0 > gens/20181129-gen-e2e-300-55-5-src_test.txt
```

11/30/18:

Train the autogressive model:

```
time python chsmm.py \
  -data data/e2e_aligned \
  -emb_size 300 \
  -hid_size 300 \
  -layers 1 \
  -K 60 \
  -L 4 \
  -log_interval 200 \
  -thresh 9 \
  -emb_drop \
  -bsz 8 \
  -max_seqlen 55 \
  -lr 0.5 \
  -sep_attn \
  -max_pool \
  -unif_lenps \
  -one_rnn \
  -Kmul 1 \
  -mlpinp \
  -onmt_decay \
  -cuda \
  -seed 1818 \
  -save models/chsmm-e2e-60-1-far.pt \
  -ar_after_decay
```

# Experiments

11/23/18:

- Ran:

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

To do the MAP segmentation:

```
python chsmm.py
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
-label_train | tee segs/seg-e2e-300-55-5.txt\
```

11/26/18:

- Ran:

To do generation:

```
time python chsmm.py \
-data data/e2e_aligned/ \
-emb_size 300 \
-hid_size 300 \
-layers 1 \
-dropout 0.3 \
-K 60 \  # this is probably wrong, since model implies: K=55, Kmul=5
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
-max_pool \
-gen_from_fi data/e2e_aligned/src_uniq_valid.txt \
-load models/chsmm-e2e-300-55-5.pt.1 \
-tagged_fi segs/seg-e2e-300-55-5.txt \
-beamsz 5 \
-ntemplates 100 \
-gen_wts '1,1' \
-cuda \
-min_gen_tokes 0 > gens/gen-e2e-300-55-5.txt
```

- Ran:

To train a dummy model locally (and step through the code):

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
  -lr 0.5 \
  -sep_attn \
  -max_pool \
  -unif_lenps \
  -one_rnn \
  -Kmul 5 \
  -mlpinp \
  -onmt_decay \
  -seed 1818 \
  -save models/DUMMY-chsmm-e2e-300-55-5.pt
```

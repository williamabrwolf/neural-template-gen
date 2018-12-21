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
-min_gen_tokes 0 > gens/20181129-gen-e2e-300-55-5-src_uniq_valid.txt
```

11/30/18:

- Ran:
  - Trained autoregressive model

```
time python chsmm.py   \
  -data data/e2e_aligned   \
  -emb_size 300   \
  -hid_size 300   \
  -layers 1   \
  -K 60   \
  -L 4   \
  -log_interval 200   \
  -thresh 9   \
  -emb_drop   \
  -bsz 8   \
  -max_seqlen 55   \
  -lr 0.5   \
  -sep_attn   \
  -max_pool   \
  -unif_lenps   \
  -one_rnn   \
  -Kmul 1   \
  -mlpinp   \
  -onmt_decay   \
  -cuda   \
  -seed 1818   \
  -save models/chsmm-e2e-60-1-far.pt\
  -ar_after_decay
```

12/3/18:

- Ran:
  - Viterbi segmentation on training data for the autoregressive model

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
  -lr 0.5  \
  -sep_attn \
  -max_pool \
  -unif_lenps \
  -one_rnn \
  -Kmul 1 \
  -mlpinp \
  -onmt_decay \
  -cuda \
  -seed 1818 \
  -load models/chsmm-e2e-60-1-far.pt.1 \
  -label_train \
  -ar_after_decay | tee segs/seg-e2e-60-1-far.txt
```

- Ran:
  - Generation on validation data for the autoregressive model

```
time python chsmm.py \
  -data data/e2e_aligned/ \
  -emb_size 300 \
  -hid_size 300 \
  -layers 1 \
  -K 60 \
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
  -Kmul 1 \
  -max_pool \
  -gen_from_fi data/e2e_aligned/src_uniq_valid.txt \
  -load models/chsmm-e2e-60-1-far.pt.1 \
  -tagged_fi segs/seg-e2e-60-1-far.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 1 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-60-1-src_uniq_valid.txt
```

---

- Ran: segmentation, generation and evaluation 5x for non-autoregressive model to assess variance in results

# run viterbi segmentation on non-autoregressive model, seed 1

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
  -seed 1 \
  -load models/chsmm-e2e-300-55-5.pt.1 \
  -label_train | tee segs/seg-e2e-300-55-5-seed-1.txt

# run viterbi segmentation on non-autoregressive model, seed 2

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
  -seed 2 \
  -load models/chsmm-e2e-300-55-5.pt.1 \
  -label_train | tee segs/seg-e2e-300-55-5-seed-2.txt

# run viterbi segmentation on non-autoregressive model, seed 3

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
  -seed 3 \
  -load models/chsmm-e2e-300-55-5.pt.1 \
  -label_train | tee segs/seg-e2e-300-55-5-seed-3.txt

# run viterbi segmentation on non-autoregressive model, seed 4

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
  -seed 4 \
  -load models/chsmm-e2e-300-55-5.pt.1 \
  -label_train | tee segs/seg-e2e-300-55-5-seed-4.txt

# run viterbi segmentation on non-autoregressive model, seed 5

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
  -seed 5 \
  -load models/chsmm-e2e-300-55-5.pt.1 \
  -label_train | tee segs/seg-e2e-300-55-5-seed-5.txt

---

# do generation on validation set with non-autoregressive model, seed 1

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
  -tagged_fi segs/seg-e2e-300-55-5-seed-1.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 1 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-55-5-src_uniq_valid-seed-1.txt

# do generation on validation set with non-autoregressive model, seed 2

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
  -tagged_fi segs/seg-e2e-300-55-5-seed-2.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 2 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-55-5-src_uniq_valid-seed-2.txt


# do generation on validation set with non-autoregressive model, seed 3

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
  -tagged_fi segs/seg-e2e-300-55-5-seed-3.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 3 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-55-5-src_uniq_valid-seed-3.txt


# do generation on validation set with non-autoregressive model, seed 4

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
  -tagged_fi segs/seg-e2e-300-55-5-seed-4.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 4 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-55-5-src_uniq_valid-seed-4.txt


# do generation on validation set with non-autoregressive model, seed 5

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
  -tagged_fi segs/seg-e2e-300-55-5-seed-5.txt \
  -beamsz 5 \
  -ntemplates 100 \
  -gen_wts '1,1' \
  -cuda \
  -seed 5 \
  -min_gen_tokes 0 > gens/20181203-gen-e2e-300-55-5-src_uniq_valid-seed-5.txt

---

# to pull down from the instances running these:

aws_pull neur-temp --region nv && aws_pull neur-temp-1 --region nv && aws_pull neur-temp-2 --region nv && aws_pull neur-temp-3 --region nv && aws_pull neur-temp-4 --region nv && aws_pull neur-temp-5 --region nv

# to push up to the instances running these:

aws_push neur-temp --region nv && aws_push neur-temp-1 --region nv && aws_push neur-temp-2 --region nv && aws_push neur-temp-3 --region nv && aws_push neur-temp-4 --region nv && aws_push neur-temp-5 --region nv

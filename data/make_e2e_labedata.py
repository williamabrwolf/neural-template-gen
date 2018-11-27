import os
import sys

from utils import get_e2e_fields, e2e_key2idx

e2e_train_src = "playground/E2E_opennmt-py/trainset-source.tok"
e2e_train_tgt = "playground/E2E_opennmt-py/trainset-target.tok"
e2e_val_src = "playground/E2E_opennmt-py/devset-source.tok"
e2e_val_tgt = "playground/E2E_opennmt-py/devset-target.tok"

punctuation = set(['.', '!', ',', ';', ':', '?'])

def get_first_sent_tokes(tokes):
    try:
        first_per = tokes.index('.')
        return tokes[:first_per+1]
    except ValueError:
        return tokes

def stupid_search(tokes, fields):
    """
    greedily assigns longest labels to spans from left to right
    """
    labels = []
    i = 0
    while i < len(tokes):
        matched = False
        for j in xrange(len(tokes), i, -1):
            # first check if it's punctuation
            if all(toke in punctuation for toke in tokes[i:j]):
                labels.append((i, j, len(e2e_key2idx))) # first label after rul labels
                i = j
                matched = True
                break
            # then check if it matches stuff in the table
            for k, v in fields.iteritems():
                # take an uncased match
                if " ".join(tokes[i:j]).lower() == " ".join(v).lower():
                    labels.append((i, j, e2e_key2idx[k]))
                    i = j
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return labels

def print_data(srcfi, tgtfi):
    with open(srcfi) as f1:
        with open(tgtfi) as f2:
            for srcline in f1:
                tgttokes = f2.readline().strip().split()
                senttokes = tgttokes

                # srcline: '__start_name__ The Vaults __end_name__ __start_eatType__ pub __end_eatType__ __start_priceRange__ more than \xc2\xa3 30 __end_priceRange__ __start_customerrating__ 5 out of 5 __end_customerrating__ __start_near__ Caf\xc3\xa9 Adriatic __end_near__\n'

                # senttokes: ['The', 'Vaults', 'pub', 'near', 'Caf\xc3\xa9', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.', 'Prices', 'start', 'at', '\xc2\xa3', '30', '.']

                fields = get_e2e_fields(srcline.strip().split()) # fieldname -> tokens

                # fields: defaultdict(<type 'list'>, {'eatType': ['pub'], 'near': ['Caf\xc3\xa9', 'Adriatic'], 'priceRange': ['more', 'than', '\xc2\xa3', '30'], 'name': ['The', 'Vaults'], 'customerrating': ['5', 'out', 'of', '5']})

                labels = stupid_search(senttokes, fields)
                labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]

                # add eos stuff
                senttokes.append("<eos>")

                labels.append((str(len(senttokes)-1), str(len(senttokes)), '8')) # label doesn't matter
                labelstr = " ".join([','.join(label) for label in labels])
                sentstr = " ".join(senttokes)

                outline = "%s|||%s" % (sentstr, labelstr)
                
                print outline


if sys.argv[1] == "train":
    print_data(e2e_train_src, e2e_train_tgt)
elif sys.argv[1] == "valid":
    print_data(e2e_val_src, e2e_val_tgt)
else:
    assert False


"""
ASAPP DOCSTRING:

E2E has 7 fields.

Iterate through natural language output, and see if the values for any fields are present, verbatim.

If found, annotate these "spans" (a start index, and an end index, *exclusive*) with the index of the key to which the value corresponds.

Annotate punctuation marks, using the `len(distinct keys) + 1` as the "index of the key."

Annotate an implicit <eos> token, using the `len(distinct keys) + 2` as the "index of the key."

For example:

Parsed structured input:

```
(Pdb) fields
defaultdict(<type 'list'>, {'eatType': ['pub'], 'near': ['Caf\xc3\xa9', 'Adriatic'], 'priceRange': ['more', 'than', '\xc2\xa3', '30'], 'name': ['The', 'Vaults'], 'customerrating': ['5', 'out', 'of', '5']})
```

Tokenized natural language output:

```
(Pdb) senttokes
['The', 'Vaults', 'pub', 'near', 'Caf\xc3\xa9', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.', 'Prices', 'start', 'at', '\xc2\xa3', '30', '.', '<eos>']
```

Key to index lookup:

```
(Pdb) e2e_key2idx
{'customerrating': 4, 'name': 0, 'area': 5, 'food': 2, 'near': 6, 'priceRange': 3, 'eatType': 1}
```

Span annotations (the thing we're computing):

```
(Pdb) labels
[('0', '2', '0'), ('2', '3', '1'), ('4', '6', '6'), ('11', '12', '7'), ('17', '18', '7'), ('18', '19', '8')]
```

*The first two tokens are the verbatim value for the 'name' key. This key has the index 0. So, we annotate as ('0', '2', '0').*

`('11', '12', '7')` indicates that we have a punctuation mark in position 11.

`('18', '19', '8')` indicates we have an (implicit) `<eos>` token in position 18.

The final output is:

```
(Pdb) outline
'The Vaults pub near Caf\xc3\xa9 Adriatic has a 5 star rating . Prices start at \xc2\xa3 30 . <eos>|||0,2,0 2,3,1 4,6,6 11,12,7 17,18,7 18,19,8'
```
"""

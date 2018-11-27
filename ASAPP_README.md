# To build the conda environment

`$ conda env create -f environment.yaml`

# Data preparation

## E2E

### Span annotations

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

### What are the different datasets?

`src_train.txt` contains the structured input.

`train.txt` contains the natural language output, with span annotations.

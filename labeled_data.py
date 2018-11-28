"""
this file modified from the word_language_model example
"""
import os
import torch

from collections import Counter, defaultdict

from data.utils import get_wikibio_poswrds, get_e2e_poswrds

import random
random.seed(1111)

#punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set() # i don't know why i was so worried about punctuation

class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>"] # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class SentenceCorpus(object):
    def __init__(self, path, bsz, thresh=0, add_bos=False, add_eos=False,
                 test=False):
        self.dictionary = Dictionary()
        self.bsz = bsz
        self.wiki = "wiki" in path

        train_src = os.path.join(path, "src_train.txt")

        if thresh > 0:
            self.get_vocabs(os.path.join(path, 'train.txt'), train_src, thresh=thresh)
            # NOTE: `ngen_types` is the size of the vocab from which we're permitted to generate
            # (excluding, notably, the things from src_path, e.g. data/e2e_aligned/src_train.txt)
            self.ngen_types = len(self.genset) + 4 # assuming didn't encounter any special tokens
            add_to_dict = False
        else:
            add_to_dict = True
        trsents, trlabels, trfeats, trlocs, inps = self.tokenize(
            os.path.join(path, 'train.txt'), train_src, add_to_dict=add_to_dict,
            add_bos=add_bos, add_eos=add_eos)
        print "using vocabulary of size:", len(self.dictionary)

        print self.ngen_types, "gen word types"
        self.train, self.train_mb2linenos = self.minibatchify(
            trsents, trlabels, trfeats, trlocs, inps, bsz) # list of minibatches

        if (os.path.isfile(os.path.join(path, 'valid.txt'))
                or os.path.isfile(os.path.join(path, 'test.txt'))):
            if not test:
                val_src = os.path.join(path, "src_valid.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'valid.txt'), val_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            else:
                print "using test data and whatnot...."
                test_src = os.path.join(path, "src_test.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'test.txt'), test_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            self.valid, self.val_mb2linenos = self.minibatchify(
                vsents, vlabels, vfeats, vlocs, vinps, bsz)


    def get_vocabs(self, path, src_path, thresh=2):
        """
        ASAPP DOCSTRING:

        E2E example:

        Example parameters:

            path: data/e2e_aligned/train.txt
            src_path: data/e2e_aligned/src_train.txt
            thresh: 9

        1. Iterate through each line in src_path (data/e2e_aligned/src_train.txt),
            split the tokens. Example:

        tokes: ['__start_name__', 'The', 'Vaults', '__end_name__', '__start_eatType__', 'pub', '__end_eatType__', '__start_priceRange__', 'more', 'than', '\xc2\xa3', '30', '__end_priceRange__', '__start_customerrating__', '5', 'out', 'of', '5', '__end_customerrating__', '__start_near__', 'Caf\xc3\xa9', 'Adriatic', '__end_near__']

        2. The tokens in `tokes` have start and end delimiters. Then, we add one (key, value) pair
        for each token in the value.

        For example, `'__start_customerrating__', '5', 'out', 'of', '5', '__end_customerrating__',`
        translates to:

        ```
        {
            ('_customerrating', 1): '5',
            ('_customerrating', 2): 'out',
            ('_customerrating', 3): 'of',
            ('_customerrating', 4): '5',
        }

        in the dict `fields`. Note that it is 1-indexed.

        3. Maintain a Counter called `tgt_voc`.

            - Update with all of the values in `fields`
            - Update with all of the string keys in `fields` ('_customerrating' in the above example)
            - Update with all of the indices in `fields` ((1, 2, 3, 4) in the above example)

        4. Maintain a list called `linewords`.

            - Append a `set` of the values in `fields` for each line to `linewords`,
            excluding punctuation

        5. Maintain a Counter called `genwords` for keep tracking of the tokens we're permitted to
            generate.

        6. Read from `path` (data/e2e_aligned/train.txt), and update `genwords` with all `words` that
            don't appear in the *corresponding line's* `linewords` set. (see Step 4).

        Presumably we do this to ensure we can't generate a token that was used in... src_train.

        7. Add all `words` to `tgt_voc`.

        8. From both `tgt_voc` and `genwords`, delete all keys that occur <= `thresh` number of times.

        9. Get all the keys in `tgt_voc`, name it `tgtkeys`, and sort in a way such that
            the key in `genwords` come first.

        10. Add everything in `tgtkeys` to an `self.dictionary` instance attribute.
        """

        """unks words occurring <= thresh times"""
        tgt_voc = Counter()
        assert os.path.exists(path)

        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd

                fieldvals = fields.values()
                tgt_voc.update(fieldvals)
                linewords.append(set(wrd for wrd in fieldvals
                                     if wrd not in punctuation))

                tgt_voc.update([k for k, idx in fields])
                tgt_voc.update([idx for k, idx in fields])

        genwords = Counter()
        with open(path, 'r') as f:
            #tokens = 0
            for l, line in enumerate(f):
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                tgt_voc.update(words)

        # prune
        # N.B. it's possible a word appears enough times in total but not in genwords
        # so we need separate unking for generation
        #print "comeon", "aerobatic" in genwords

        for cntr in [tgt_voc, genwords]:
            for k in cntr.keys():
                if cntr[k] <= thresh:
                    del cntr[k]

        self.genset = set(genwords.keys())
        tgtkeys = tgt_voc.keys()
        # make sure gen stuff is first
        tgtkeys.sort(key=lambda x: -(x in self.genset))
        self.dictionary.bulk_add(tgtkeys)
        # make sure we did everything right (assuming didn't encounter any special tokens)

        # TODO: I don't know what this 4 is
        assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset
        self.dictionary.add_word("<ncf1>", train=True)  # `ncf` = "not copied from"
        self.dictionary.add_word("<ncf2>", train=True)  # `ncf` = "not copied from"
        self.dictionary.add_word("<ncf3>", train=True)  # `ncf` = "not copied from"
        self.dictionary.add_word("<go>", train=True)
        self.dictionary.add_word("<stop>", train=True)


    def tokenize(self, path, src_path, add_to_dict=False, add_bos=False, add_eos=False):
        """
        ASAPP DOCSTRING:

        This method does a few things:

        1. Iterate through entries in the source (the structured input), and compute:

        `feats`:

            Converting this source into `fields`:

            (Pdb) fields

            fields = {
                ('_customerrating', 1): '5',
                ('_customerrating', 2): 'out',
                ('_customerrating', 3): 'of',
                ('_customerrating', 4): '5',
                ('_near', 1): 'Caf\xc3\xa9',
                ('_near', 2): 'Adriatic',
                ('_name', 1): 'The',
                ('_name', 2): 'Vaults',
                ('_priceRange', 1): 'more',
                ('_priceRange', 2): 'than',
                ('_priceRange', 3): '\xc2\xa3',
                ('_priceRange', 4): '30',
                ('_eatType', 1): 'pub'
            }

            For each `word` in the values of `fields`, compute:

            1. The global numerical index of the string in the key (e.g. '_customerrating')
            2. The global numerical index of the idx in the key (e.g. 1)
            3. The global numerical index of the `word` itself

            To `feats`, append a list of these 3 items.

            The list of individual `feats`--one for each record in the input--are put into
            a final list called:

                `src_feats`

        `wrd2idxs`:

            For each `word` in the values of `fields`, compute:

            The index *in the iteration* at which it occurs.

            Were we to iterate in the order of the sentence, this would just be the
            index of the word in this sentence (where words can have multiple indices,
            if they appear more than once!).

            As is, it's the index wherein we iterate in unsorted order, namely via
            iteration through the values of a dict.

            To `wrd2idxs[word]`, append this index. We append, because again, a given
            word can have more than one index if it occurs more than once in the values
            for this record's `fields`.

        `wrd2fields`:

            For each `word` in the values of `fields`, compute:

            `(word global index, key global index, idx global index, cheatfeat index)`

            The first three are plucked from its features, which we appended to `feats`, above.

            The last item is defined as:

            "Is this the final value you are procesing for a given key?

            For example, _customerrating has 4 keys:

            ('_customerrating', 1): '5',
            ('_customerrating', 2): 'out',
            ('_customerrating', 3): 'of',
            ('_customerrating', 4): '5',

            This item asks: are we processing the 4th?

            `cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]`

        2. Iterate through words in the target (the natural language output), and compute:

        `sent`:

            A list of the global numerical indices for each word,

            *but only for the words that are in the `self.genset`, i.e. did not
            occur in the values of its corresponding `fields` (the dict that houses
            its structured input data).*

            If this word is not in `self.genset`, i.e. it was "seen before", we use the

            global numerical index of "<unk>".

        `copied`:

            A list of:

                If:

                    The word was in the values of `fields`, i.e.

                        - It was "seen before"
                        - It is not in `self.genset`

                Then: append the `wrd2idxs` index of this word.

                Else: Use -1 as the index.

            Almost conversely, this is maintaining metadata about those words that were "copied."

        `insent`:

            A list of:

                If:

                    The word was in the values of `fields`, i.e.

                        - It was "seen before"
                        - It is not in `self.genset`

                Then: the `wrd2fields` list for this word.

                Else: [global numerical index for word, w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]]

                ...where the latter 3 elements are constants.

            This should similarly be considered metadata about the items that were copied.

        `labelist`:

            A newly tuple-ified span annotation, e.g.

            [(0, 2, 0), (2, 3, 1), (4, 6, 6), (11, 12, 7), (17, 18, 7), (18, 19, 8)]

        The final data structures containing the previous 4 structures, for each of the words
        in the natural language output, are named:

            `sents, copylocs, inps, labels`

        :return: sents, labels, src_feats, copylocs, inps

        """

        """Assumes fmt is sentence|||s1,e1,k1 s2,e2,k2 ...."""
        assert os.path.exists(path)

        src_feats, src_wrd2idxs, src_wrd2fields = [], [], []
        w2i = self.dictionary.word2idx
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                #fields = get_e2e_fields(tokes, keys=self.e2e_keys) #keyname -> list of words
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd

                # wrd2things will be unordered
                feats, wrd2idxs, wrd2fields = [], defaultdict(list), defaultdict(list)
                # get total number of words per field
                fld_cntr = Counter([key for key, _ in fields])
                for (k, idx), wrd in fields.iteritems():
                    if k in w2i:
                        """
                        (Pdb) k
                        '_customerrating'
                        (Pdb) idx
                        1
                        (Pdb) wrd
                        '5'
                        """
                        featrow = [self.dictionary.add_word(k, add_to_dict),
                                   self.dictionary.add_word(idx, add_to_dict),
                                   self.dictionary.add_word(wrd, add_to_dict)]
                        """
                        (Pdb) featrow
                        [861, 780, 400]
                        """

                        wrd2idxs[wrd].append(len(feats))

                        #nflds = self.dictionary.add_word(fld_cntr[k], add_to_dict)

                        cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]
                        """
                        Append: (word global index, key global index, idx global index, cheatfeat index)
                        """
                        wrd2fields[wrd].append((featrow[2], featrow[0], featrow[1], cheatfeat))
                        feats.append(featrow)
                src_wrd2idxs.append(wrd2idxs)
                src_wrd2fields.append(wrd2fields)
                src_feats.append(feats)

        sents, labels, copylocs, inps = [], [], [], []

        # Add words to the dictionary
        tgtline = 0
        with open(path, 'r') as f:
            #tokens = 0
            for line in f:
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                sent, copied, insent = [], [], []
                if add_bos:
                    sent.append(self.dictionary.add_word('<bos>', True))
                for word in words:
                    # sent is just used for targets; we have separate inputs
                    """
                    If we didn't omit this previously, i.e. *it does not appear in the values
                    of its structured-input.*
                    """
                    if word in self.genset:
                        sent.append(w2i[word])
                    else:
                        """
                        This word did appear in its corresponding `fields` values; we replace
                        it with "<unk>".
                        """
                        sent.append(w2i["<unk>"])
                    if word not in punctuation and word in src_wrd2idxs[tgtline]:
                        """
                        The indices of the words that were "copied over" from input to output.
                        """
                        copied.append(src_wrd2idxs[tgtline][word])
                        """
                        You'll have multiple 4-tuples if this word appeared more that once in
                        the structured input.
                        """
                        winps = [[widx, kidx, idxidx, nidx]
                                 for widx, kidx, idxidx, nidx in src_wrd2fields[tgtline][word]]
                        insent.append(winps)
                        """
                        In the genset, i.e. not in the corresponding structured input's values.
                        """
                    else:
                        #assert sent[-1] < self.ngen_types
                        copied.append([-1])
                         # 1 x wrd, tokennum, totalnum
                        #insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"]]])
                        """
                        sent[-1] gives the global word index for this word.
                        """
                        insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]]])
                #sent.extend([self.dictionary.add_word(word, add_to_dict) for word in words])
                if add_eos:
                    sent.append(self.dictionary.add_word('<eos>', True))
                labetups = [tupstr.split(',') for tupstr in spanlabels.split()]
                labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]

                """
                (Pdb) sent
                [503, 0, 471, 144, 407, 0, 142, 450, 400, 755, 310, 136, 126, 391, 399, 772, 0, 136, 255]

                (Pdb) labelist
                [(0, 2, 0), (2, 3, 1), (4, 6, 6), (11, 12, 7), (17, 18, 7), (18, 19, 8)]

                (Pdb) copied
                [[7], [2], [12], [-1], [3], [1], [-1], [-1], [0, 5], [-1], [-1], [-1], [-1], [-1], [-1], [10], [4], [-1], [-1]]

                (Pdb) insent
                [[[503, 855, 780, 870]], [[840, 855, 781, 871]], [[471, 865, 780, 871]], [[144, 867, 868, 869]], [[407, 850, 780, 870]], [[820, 850, 781, 871]], [[142, 867, 868, 869]], [[450, 867, 868, 869]], [[400, 861, 780, 870], [400, 861, 783, 871]], [[755, 867, 868, 869]], [[310, 867, 868, 869]], [[136, 867, 868, 869]], [[126, 867, 868, 869]], [[391, 867, 868, 869]], [[399, 867, 868, 869]], [[772, 814, 782, 870]], [[854, 814, 783, 871]], [[136, 867, 868, 869]], [[255, 867, 868, 869]]]
                """
                sents.append(sent)
                labels.append(labelist)
                copylocs.append(copied)
                inps.append(insent)
                tgtline += 1
        assert len(sents) == len(labels)
        assert len(src_feats) == len(sents)
        assert len(copylocs) == len(sents)
        return sents, labels, src_feats, copylocs, inps

    def featurize_tbl(self, fields):
        """
        fields are key, pos -> wrd maps
        returns: nrows x nfeats tensor
        """
        feats = []
        for (k, idx), wrd in fields.iteritems():
            if k in self.dictionary.word2idx:
                featrow = [self.dictionary.add_word(k, False),
                           self.dictionary.add_word(idx, False),
                           self.dictionary.add_word(wrd, False)]
                feats.append(featrow)
        return torch.LongTensor(feats)

    def padded_loc_mb(self, curr_locs):
        """
        curr_locs is a bsz-len list of tgt-len list of locations
        returns:
          a seqlen x bsz x max_locs tensor
        """
        max_locs = max(len(locs) for blocs in curr_locs for locs in blocs)
        for blocs in curr_locs:
            for locs in blocs:
                if len(locs) < max_locs:
                    locs.extend([-1]*(max_locs - len(locs)))
        return torch.LongTensor(curr_locs).transpose(0, 1).contiguous()

    def padded_feat_mb(self, curr_feats):
        """
        curr_feats is a bsz-len list of nrows-len list of features
        returns:
          a bsz x max_nrows x nfeats tensor
        """
        max_rows = max(len(feats) for feats in curr_feats)
        nfeats = len(curr_feats[0][0])
        for feats in curr_feats:
            if len(feats) < max_rows:
                [feats.append([self.dictionary.word2idx["<pad>"] for _ in xrange(nfeats)])
                 for _ in xrange(max_rows - len(feats))]
        return torch.LongTensor(curr_feats)


    def padded_inp_mb(self, curr_inps):
        """
        curr_inps is a bsz-len list of seqlen-len list of nlocs-len list of features
        returns:
          a bsz x seqlen x max_nlocs x nfeats tensor
        """
        max_rows = max(len(feats) for seq in curr_inps for feats in seq)
        nfeats = len(curr_inps[0][0][0])
        for seq in curr_inps:
            for feats in seq:
                if len(feats) < max_rows:
                    # pick random rows
                    randidxs = [random.randint(0, len(feats)-1)
                                for _ in xrange(max_rows - len(feats))]
                    [feats.append(feats[ridx]) for ridx in randidxs]
        return torch.LongTensor(curr_inps)


    def minibatchify(self, sents, labels, feats, locs, inps, bsz):
        """
        ASAPP DOCSTRING:

        (Pdb) type(sents)
        <type 'list'>
        (Pdb) len(sents)
        42061
        (Pdb) sents[0]
        [503, 0, 471, 144, 407, 0, 142, 450, 400, 755, 310, 136, 126, 391, 399, 772, 0, 136, 255]

        (Pdb) type(labels)
        <type 'list'>
        (Pdb) len(labels)
        42061
        (Pdb) labels[0]
        [(0, 2, 0), (2, 3, 1), (4, 6, 6), (11, 12, 7), (17, 18, 7), (18, 19, 8)]

        (Pdb) type(feats)
        <type 'list'>
        (Pdb) len(feats)
        42061
        (Pdb) feats[0]
        [[861, 780, 400], [850, 781, 820], [855, 781, 840], [850, 780, 407], [814, 783, 854], [861, 783, 400], [861, 781, 672], [855, 780, 503], [814, 780, 433], [861, 782, 382], [814, 782, 772], [814, 781, 293], [865, 780, 471]]

        (Pdb) type(locs)
        <type 'list'>
        (Pdb) len(locs)
        42061
        (Pdb) locs[0]
        [[7], [2], [12], [-1], [3], [1], [-1], [-1], [0, 5], [-1], [-1], [-1], [-1], [-1], [-1], [10], [4], [-1], [-1]]

        (Pdb) type(inps)
        <type 'list'>
        (Pdb) len(inps)
        42061
        (Pdb) inps[0]
        [[[503, 855, 780, 870]], [[840, 855, 781, 871]], [[471, 865, 780, 871]], [[144, 867, 868, 869]], [[407, 850, 780, 870]], [[820, 850, 781, 871]], [[142, 867, 868, 869]], [[450, 867, 868, 869]], [[400, 861, 780, 870], [400, 861, 783, 871]], [[755, 867, 868, 869]], [[310, 867, 868, 869]], [[136, 867, 868, 869]], [[126, 867, 868, 869]], [[391, 867, 868, 869]], [[399, 867, 868, 869]], [[772, 814, 782, 870]], [[854, 814, 783, 871]], [[136, 867, 868, 869]], [[255, 867, 868, 869]]]
        """

        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """
        # sort in ascending order
        """
        `sents` are sorted in increasing order of length; sorted_idxs contains their corresponding index in the original `sents` input.

        (This might seem dumb. Presumably we'll use this information momentarily to index into the other equally-sized inputs.)
        """
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []
        curr_batch, curr_labels, curr_feats, curr_locs, curr_linenos = [], [], [], [], []
        curr_inps = []
        curr_len = len(sents[0])
        for i in xrange(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz: # we're done
                """
                This is kind of neat:

                Append, i.e. finalize the batch, if you reach the batch size, or:

                if the current sentence is not the same length as the previous sentence. (This way,
                as they say, there is never any reason for padding!)
                """

                """
                The first lists comprised of > 1 elements (i.e. they had > 1 corresponding sentences of the same length).

                Before any padding (on the lists that were not `curr_batch`, i.e. the ones containing things that aren't
                the global numerical word indices that comprise a sentence):

                (Pdb) curr_batch
                [[253, 533, 632, 0, 136, 255], [0, 516, 56, 139, 136, 255], [396, 307, 533, 147, 0, 255], [503, 0, 142, 158, 461, 255], [0, 533, 147, 141, 619, 255]]

                (Pdb) curr_labels
                [[(0, 1, 4), (1, 2, 1), (3, 4, 0), (4, 5, 7), (5, 6, 8)], [(0, 1, 0), (4, 5, 7), (5, 6, 8)], [(1, 2, 4), (2, 3, 1), (4, 5, 0), (5, 6, 8)], [(0, 2, 0), (3, 4, 2), (5, 6, 8)], [(0, 1, 0), (1, 2, 1), (5, 6, 8)]]

                (Pdb) curr_feats
                [[[865, 780, 533], [861, 780, 307], [855, 780, 846]], [[865, 780, 533], [862, 780, 799], [855, 780, 810]], [[865, 780, 533], [861, 780, 307], [855, 780, 846]], [[803, 780, 158], [861, 780, 307], [855, 781, 791], [855, 780, 503]], [[865, 780, 533], [862, 780, 799], [855, 780, 810]]]

                (Pdb) curr_locs
                [[[-1], [0], [-1], [2], [-1], [-1]], [[2], [-1], [-1], [-1], [-1], [-1]], [[-1], [1], [0], [-1], [2], [-1]], [[3], [2], [-1], [0], [-1], [-1]], [[2], [0], [-1], [-1], [-1], [-1]]]

                (Pdb) curr_inps
                [[[[253, 867, 868, 869]], [[533, 865, 780, 871]], [[632, 867, 868, 869]], [[846, 855, 780, 871]], [[136, 867, 868, 869]], [[255, 867, 868, 869]]], [[[810, 855, 780, 871]], [[516, 867, 868, 869]], [[56, 867, 868, 869]], [[139, 867, 868, 869]], [[136, 867, 868, 869]], [[255, 867, 868, 869]]], [[[396, 867, 868, 869]], [[307, 861, 780, 871]], [[533, 865, 780, 871]], [[147, 867, 868, 869]], [[846, 855, 780, 871]], [[255, 867, 868, 869]]], [[[503, 855, 780, 870]], [[791, 855, 781, 871]], [[142, 867, 868, 869]], [[158, 803, 780, 871]], [[461, 867, 868, 869]], [[255, 867, 868, 869]]], [[[810, 855, 780, 871]], [[533, 865, 780, 871]], [[147, 867, 868, 869]], [[141, 867, 868, 869]], [[619, 867, 868, 869]], [[255, 867, 868, 869]]]]

                Now, in the form in which they're appended to `minibatches`:

                (Pdb) torch.LongTensor(curr_batch).t().contiguous()
                 253    0  396  503    0
                 533  516  307    0  533
                 632   56  533  142  147
                   0  139  147  158  141
                 136  136    0  461  619
                 255  255  255  255  255
                [torch.LongTensor of size 6x5]

                (Pdb) curr_labels
                [[(0, 1, 4), (1, 2, 1), (3, 4, 0), (4, 5, 7), (5, 6, 8)], [(0, 1, 0), (4, 5, 7), (5, 6, 8)], [(1, 2, 4), (2, 3, 1), (4, 5, 0), (5, 6, 8)], [(0, 2, 0), (3, 4, 2), (5, 6, 8)], [(0, 1, 0), (1, 2, 1), (5, 6, 8)]]

                (Pdb) self.padded_feat_mb(curr_feats)
                (0 ,.,.) =
                  865  780  533
                  861  780  307
                  855  780  846
                    1    1    1

                (1 ,.,.) =
                  865  780  533
                  862  780  799
                  855  780  810
                    1    1    1

                (2 ,.,.) =
                  865  780  533
                  861  780  307
                  855  780  846
                    1    1    1

                (3 ,.,.) =
                  803  780  158
                  861  780  307
                  855  781  791
                  855  780  503

                (4 ,.,.) =
                  865  780  533
                  862  780  799
                  855  780  810
                    1    1    1
                [torch.LongTensor of size 5x4x3]

                (Pdb) self.padded_loc_mb(curr_locs)
                (0 ,.,.) =
                 -1
                  2
                 -1
                  3
                  2

                (1 ,.,.) =
                  0
                 -1
                  1
                  2
                  0

                (2 ,.,.) =
                 -1
                 -1
                  0
                 -1
                 -1

                (3 ,.,.) =
                  2
                 -1
                 -1
                  0
                 -1

                (4 ,.,.) =
                 -1
                 -1
                  2
                 -1
                 -1

                (5 ,.,.) =
                 -1
                 -1
                 -1
                 -1
                 -1
                [torch.LongTensor of size 6x5x1]

                (Pdb) self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()

                ...

                533  865  780  871

                (2 ,3 ,.,.) =
                142  867  868  869

                (2 ,4 ,.,.) =
                147  867  868  869

                (3 ,0 ,.,.) =
                846  855  780  871

                (3 ,1 ,.,.) =
                139  867  868  869

                (3 ,2 ,.,.) =
                147  867  868  869

                (3 ,3 ,.,.) =
                158  803  780  871

                (3 ,4 ,.,.) =
                141  867  868  869

                (4 ,0 ,.,.) =
                136  867  868  869

                (4 ,1 ,.,.) =
                136  867  868  869

                (4 ,2 ,.,.) =
                846  855  780  871

                (4 ,3 ,.,.) =
                461  867  868  869

                (4 ,4 ,.,.) =
                619  867  868  869

                (5 ,0 ,.,.) =
                255  867  868  869

                (5 ,1 ,.,.) =
                255  867  868  869

                (5 ,2 ,.,.) =
                255  867  868  869

                (5 ,3 ,.,.) =
                255  867  868  869

                (5 ,4 ,.,.) =
                255  867  868  869
                [torch.LongTensor of size 6x5x1x4]
                """
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [labels[sorted_idxs[i]]]
                curr_feats = [feats[sorted_idxs[i]]]
                curr_locs = [locs[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])

                """
                After the first append:

                (Pdb) curr_batch
                [[407, 255]]
                (Pdb) curr_labels
                [[(1, 2, 8)]]
                (Pdb) curr_feats
                [[[803, 780, 229], [850, 781, 808], [850, 780, 503], [850, 782, 800], [855, 780, 858], [814, 780, 379]]]
                (Pdb) curr_locs
                [[[-1], [-1]]]
                (Pdb) curr_inps
                [[[[407, 867, 868, 869]], [[255, 867, 868, 869]]]]
                (Pdb) curr_linenos
                [2010]
                """
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                curr_labels, self.padded_feat_mb(curr_feats),
                                self.padded_loc_mb(curr_locs),
                                self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

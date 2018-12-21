# Flow Induction: Outline

The high level goal of this project is to ingest historical conversation data and produce sequences of meaningful latent states $z$ which will be useful for downstream tasks

1. summarization at a corpus level (flow)
2. summarization at a conversation level (depositions)
3. goal-oriented dialog: select the next state given the previous states and context, in order to maximize some reward (issue resolved, business KPIs)


### Related Work
To put this project in terms of related work, we want to make a structured (optionally semi-supervised) version of [2], or a dialogue version of [1], or a language model version of [3]


### Model

We decompose the task into 2 main components

#### Language Model

Model is essentially a modified seq2seq model. Given the previous state $z_{t-1}$, previous utterances $x_{t-1}$, and other context from the conversation $c$, predict the next state $z_{t}$ and $x_t$. Then the sequence of $z=\{z_{t_1}, \ldots z_{t_T}\}$ can be read off and fed to the next model. 


In the case of [2], they want to decouple states from context. This happens at inference time with the following decomposition:
<!--$x \rightarrow q(z | x) \rightarrow$-->

- Encoder $\mathcal{R}(z | x)$
- Contextual Decoder $c \rightarrow \pi(z | c) \rightarrow \mathcal{F}(x | z,c)$



<!-- - $\mathcal{R}$: an RNN the encodes $\mathbf{x}$ into $\mathbf{z}$
	- NB: does not depend on $\mathbf{c}$ so as to learn "context-independent" semantics
- $\mathcal{F}^e$: an RNN that encodes $\mathbf{c}$ into $h^e$
- $\pi$: a network that learns $\pi(\mathbf{z}\vert\mathbf{c})$
- $\mathcal{F}^d$: an RNN that predicts $\tilde{\mathbf{x}} = \mathcal{F}^d(\tilde{\mathbf{x}}\vert \mathbf{z} \sim \pi(\mathbf{z}\vert\mathbf{c}), h^e)$
chrome extension for rendering latex in github with mathjax (single $ $) https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related
-->

Our $z$'s can and should make use of structure inherent in customer service dialogue. For example, we can enforce separate states for customer and agent, or separate sets of states for "phases" of a conversation inbetween breakpoints.

We can also leverage limited, noisy knowledge of $z$ during training. Rather than purely unsupervised, we can consider a semisupervised setup where some examples are ground truth pairs ($x$, $z$). Context may be given as additional $z$ dimensions. Here are some potential sources of context. Many of these would be more useful for product than pure research:

- CSRS intent
- SRU output from CSRS model
- Sentiment 
- Customer/agent specific features
- Aggregate transition probabilities $p(z_t = z_j | z_{t-1}=z_i )$.




#### Summarization Model
The neural template model in [1] uses a simple, baseline summarization step: for generation, only use the top 100 most common sequences of states (templates)

Propose to use the following summarization framework, extending Michael Griffiths's previous setup:

1. Choose a similarity function between 2 sequences of states, e.g. Frechet distance, edit distance, weighted edit distance with (prespecified/learned) weights, (prespecified/learned polynomial) kernel distance function. One option is to weight the beginning and end of the conversations to have lower distance penalty than the middle, to encourage diversity multiple diverse sentences which start and end the same way. These are all essentially hyperparameters. Another option is to consider similarity of n-gram subsequences of $z$, so that sequences are similar if they contain similar blocks of states.
2. Solve a clustering or facility location style objective function to produce candidate flows and example conversations which they represent
3. Compute train and dev metrics on a subset of intents, check that choice of hyperparameters/model design generalizes on a holdout set of different intents


### Evaluation

- Purity (from [1]): each record/tag has a ground truth set of words associated with it. For example, the City record may have New York, Buenos Aires sample tokens conditioned on $z$ and check how well it correlates with a training tag. $p(x | z_i)$ New York, Main Street, Buenos Aires, Canada. Quantitative as well as quantitative

- Perplexity, BLEU, ROGUE scores, etc. to measure how well $z$ can be used to predict/recover $x$. Quantitative

- Issue coverage (optimization objective and online metric for number of issues sent to an algorithmic flow). Quantitative

- Solely qualitative: do product/deployment teams find flow candidates useful, accelerate development of new flows

### Data

Public Benchmark Datasets

- E2E Challenge
- Dialogue State Tracking Challenge (ground truth states)
- Stanford Multi-Domain Dialog (SMD)
- Daily Dialog (DD)
- ReDial: movie recommendation dialogue dataset where one MTurker recommends movies to another MTurker based on their preferences [4] 
<details><summary>example</summary><p>
>
{'movieMentions': {'203371': 'Final Fantasy: The Spirits Within (2001)',
  '84779': 'The Triplets of Belleville (2003)',
  '122159': 'Mary and Max (2009)',
  '151313': 'A Scanner Darkly  (2006)',
  '191602': 'Waking Life (2001)',
  '165710': 'The Boss Baby (2017)'},
 'respondentQuestions': {'203371': {'suggested': 1, 'seen': 0, 'liked': 1},
  '84779': {'suggested': 0, 'seen': 1, 'liked': 1},
  '122159': {'suggested': 0, 'seen': 1, 'liked': 1},
  '151313': {'suggested': 0, 'seen': 1, 'liked': 1},
  '191602': {'suggested': 0, 'seen': 1, 'liked': 1},
  '165710': {'suggested': 1, 'seen': 0, 'liked': 1}},
 'messages': [{'timeOffset': 0,
   'text': "Hi there, how are you? I'm looking for movie recommendations",
   'senderWorkerId': 0,
   'messageId': 1021},
  {'timeOffset': 15,
   'text': 'I am doing okay. What kind of movies do you like?',
   'senderWorkerId': 1,
   'messageId': 1022},
  {'timeOffset': 66,
   'text': 'I like animations like @84779 and @191602',
   'senderWorkerId': 0,
   'messageId': 1023},
  {'timeOffset': 86,
   'text': 'I also enjoy @122159',
   'senderWorkerId': 0,
   'messageId': 1024},
  {'timeOffset': 95,
   'text': 'Anything artistic',
   'senderWorkerId': 0,
   'messageId': 1025},
  {'timeOffset': 135,
   'text': 'You might like @165710 that was a good movie.',
   'senderWorkerId': 1,
   'messageId': 1026},
  {'timeOffset': 151,
   'text': "What's it about?",
   'senderWorkerId': 0,
   'messageId': 1027},
  {'timeOffset': 207,
   'text': 'It has Alec Baldwin it is about a baby that works for a company and gets adopted it is very funny',
   'senderWorkerId': 1,
   'messageId': 1028},
  {'timeOffset': 238,
   'text': 'That seems like a nice comedy',
   'senderWorkerId': 0,
   'messageId': 1029},
  {'timeOffset': 272,
   'text': 'Do you have any animated recommendations that are a bit more dramatic? Like @151313 for example',
   'senderWorkerId': 0,
   'messageId': 1030},
  {'timeOffset': 327,
   'text': 'I like comedies but I prefer films with a little more depth',
   'senderWorkerId': 0,
   'messageId': 1031},
  {'timeOffset': 467,
   'text': 'That is a tough one but I will remember something',
   'senderWorkerId': 1,
   'messageId': 1032},
  {'timeOffset': 509,
   'text': '@203371 was a good one',
   'senderWorkerId': 1,
   'messageId': 1033},
  {'timeOffset': 564,
   'text': "Ooh that seems cool! Thanks for the input. I'm ready to submit if you are.",
   'senderWorkerId': 0,
   'messageId': 1034},
  {'timeOffset': 571,
   'text': 'It is animated, sci fi, and has action',
   'senderWorkerId': 1,
   'messageId': 1035},
  {'timeOffset': 579,
   'text': 'Glad I could help',
   'senderWorkerId': 1,
   'messageId': 1036},
  {'timeOffset': 581, 'text': 'Nice', 'senderWorkerId': 0, 'messageId': 1037},
  {'timeOffset': 591,
   'text': 'Take care, cheers!',
   'senderWorkerId': 0,
   'messageId': 1038},
  {'timeOffset': 608, 'text': 'bye', 'senderWorkerId': 1, 'messageId': 1039}],
 'conversationId': '391',
 'respondentWorkerId': 1,
 'initiatorWorkerId': 0,
 'initiatorQuestions': {'203371': {'suggested': 1, 'seen': 0, 'liked': 1},
  '84779': {'suggested': 0, 'seen': 1, 'liked': 1},
  '122159': {'suggested': 0, 'seen': 1, 'liked': 1},
  '151313': {'suggested': 0, 'seen': 1, 'liked': 1},
  '191602': {'suggested': 0, 'seen': 1, 'liked': 1},
  '165710': {'suggested': 1, 'seen': 0, 'liked': 1}}}
</p></details>

- Auction: dataset with labeled examples of policies/utilities for both parties (+ outcome). One example could be the [Craigslist Bargain dataset](https://stanfordnlp.github.io/cocoa/)
<details><summary>example</summary><p>
>JVC HD-ILA 1080P 70 Inch TV (\$275)
> Tv is approximately 10 years old. Just installed  new lamp. There are 2 HDMI inputs. Works and looks like new.
>
> A: Hello 
> B: Hello there
> A: So, are you interested in this great TV? Honestly, I barely used it and decided to sell it because I don’t really watch much TV these days. I’m selling it for \$275
> B: I am definitely interested in the TV, but it being 10 years old has me a bit skeptical. How does the TV look running movies and games, if you don’t mind me asking.
> A: It’s full HD at 1080p and it looks great. The TV works like it is brand new. I’ll throw in a DVD player that was hooked up to it for the same price of \$275
> B: The DVD player sounds nice, but unfortunately I’m on somewhat of a budget. Would you be willing to drop the price a tad, maybe \$230?
> A: \$230 is kind of low. I’ll tell ya what, if you come pick it up where it is located I’ll sell it for \$260
> B: Throw in a couple of movies with that DVD player,and you have yourself a deal.
> A: Deal.
> B: OFFER \$260.00
> A: ACCEPT
</p></details>

Production datasets

-  Troubleshooting task: Chats based around a troubleshooting tree. Annotators were given the tree and prompts at each node which correspond to meaningful states tag, and then supplied responses which fit the prompt. `agent` text + `agent_abstract` tag + `user` text (no states) 
<details><summary>examples</summary><p>
>  [{"dialogue": [{"state": "1", "sentence": "AGENT: Hi, how can I be of assistance?"}, {"state": "2", "sentence": "USER: Physical damage to my device."}, {"state": "2", "sentence": "AGENT: Do you have protection insurance?"}, {"state": "3", "sentence": "USER: I have Apple Care."}, {"state": "3", "sentence": "AGENT: <This is your solution>"}, {"state": "147", "sentence": "USER: <End Conversation>"}, {"state": "147", "sentence": "AGENT: \<End Conversation\>"}]
>
>  "user": "Physical damage to my device.", 
>            "agent": "Do you have device protection insurance on this device?", 
>            "agent_abstract": "ASK protection insurance", 
>            "slot": "protection insurance",
>            "node": [ 
>            ...
>            ]
</p></details>

- Agent summary data: conversation text + regex tags coming from conversation summary (conversation level, not sentence level)
- Future tooling events: During flow construction, metadata associated with flows could be used to train flow induction models. For example, the IDs of certain AIDL blocks could be useful state information


Ideally we can train on a mixture of several data sources and see what model parameters generalize across different dialog domains


### Current Progress
Baseline implementations of [1] and [2], the latter built from scratch in PyTorch 1.0.


## References
- [1] Neural Template Generation [code](https://github.com/harvardnlp/neural-template-gen)
[paper](https://arxiv.org/abs/1808.10122)

- [2] Unsupervised Discrete Sentence Representation Learning for Interpretable Neural Dialog Generation [code](https://github.com/snakeztc/NeuralDialog-LAED)
[paper](https://arxiv.org/abs/1804.08069)

- [3] Dynamic Topic Models [paper](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf)

- [4] Towards Deep Conversational Recommendations [ReDial dataset](https://github.com/ReDialData/website/tree/data)
[paper](http://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.pdf)

[Project repo](https://github.com/ASAPPinc/research/tree/master/will/flow-induction)

[Neural Template Fork](https://github.com/williamabrwolf/neural-template-gen)

Additional Reading List

- https://arxiv.org/abs/1811.01012
- https://arxiv.org/abs/1811.00135
- https://arxiv.org/abs/1811.01135
- https://arxiv.org/abs/1802.04942
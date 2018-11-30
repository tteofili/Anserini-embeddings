# Anserini-embeddings

Anserini utilities for working with word embeddings.
Currently, these tools are held in a separate repository because these are experimental features that depend on [deeplearning4j](https://deeplearning4j.org/), and direct inclusion of all dependent artifacts in Anserini would blow up the size of the Anserini fatjar.

Here's a sample invocation of taking GloVe embeddings and creating a Lucene index for lookup.
This is treating Lucene as a simple key-value store.

```
$ target/appassembler/bin/IndexGloVe -index glove -input glove.840B.300d.txt
```

Simple lookup example:

```
$ target/appassembler/bin/LookupGloVe -index glove-float -word "happy"
```

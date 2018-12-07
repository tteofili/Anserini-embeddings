# Anserini-embeddings

Anserini utilities for working with word embeddings.
Currently, these tools are held in a separate repository because these are experimental features that depend on [deeplearning4j](https://deeplearning4j.org/), and direct inclusion of all dependent artifacts in Anserini would blow up the size of the Anserini fatjar.

Here's a sample invocation of taking GloVe embeddings and creating a Lucene index for lookup.
This is treating Lucene as a simple key-value store.

```
$ target/appassembler/bin/IndexWordEmbeddings -index glove -input glove.840B.300d.txt
```

Simple lookup example:

```
$ target/appassembler/bin/LookupWordEmbeddings -index glove -word "happy"
```

Nearest neighbour search
========================

Index dimensionality reduced word embeddings as Lucene `FloatPoints`. 
Number of dimensions allowed is between 1 and 8.

```
$ target/appassembler/bin/IndexReducedWordEmbeddings -index glove -input glove.840B.300d.txt -dimensions 8
```

Simple nearest neighbour example:

```
$ target/appassembler/bin/NearestNeighbour -index glove -word "thomas"
thomas
vincent
lewis
fred
williams
```

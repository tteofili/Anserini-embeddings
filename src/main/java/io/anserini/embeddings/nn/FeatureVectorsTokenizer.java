package io.anserini.embeddings.nn;

import org.apache.lucene.analysis.util.CharTokenizer;

/**
 * {@link CharTokenizer} which splits at whitespaces and commas
 */
public class FeatureVectorsTokenizer extends CharTokenizer {
    @Override
    protected boolean isTokenChar(int c) {
      char c1 = Character.toChars(c)[0];
      return c1 != ',' && !Character.isWhitespace(c);
    }
  }
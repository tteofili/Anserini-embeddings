package io.anserini.embeddings.nn.vnlsh;

import io.anserini.embeddings.nn.FeatureVectorsTokenizer;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.minhash.MinHashFilter;
import org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;

/**
 * {@link Analyzer} for vector n-grams LSH based search
 */
public class VectorNgramLSHAnalyzer extends Analyzer {

  private static final int DEFAULT_SHINGLE_SIZE = 5;
  private static final int DEFAULT_DECIMALS = 1;

  private final int min;
  private final int max;
  private final int hashCount;
  private final int bucketCount;
  private final int hashSetSize;
  private final int decimals;

  private VectorNgramLSHAnalyzer(int min, int max, int hashCount, int bucketCount, int hashSetSize, int decimals) {
    super();
    this.min = min;
    this.max = max;
    this.hashCount = hashCount;
    this.bucketCount = bucketCount;
    this.hashSetSize = hashSetSize;
    this.decimals = decimals;
  }

  public VectorNgramLSHAnalyzer() {
    this(DEFAULT_SHINGLE_SIZE, DEFAULT_SHINGLE_SIZE, MinHashFilter.DEFAULT_HASH_COUNT, MinHashFilter.DEFAULT_BUCKET_COUNT,
            MinHashFilter.DEFAULT_HASH_SET_SIZE, DEFAULT_DECIMALS);
  }

  public VectorNgramLSHAnalyzer(int decimals, int ngrams, int hashCount, int bucketCount, int hashSetSize) {
    this(ngrams, ngrams, decimals, hashCount, bucketCount, hashSetSize);
  }

  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    Tokenizer source = new FeatureVectorsTokenizer();
    TokenFilter truncate = new TruncateTokenFilter(source, decimals);
    TokenFilter featurePos = new FeaturePositionTokenFilter(truncate);
    TokenStream filter;
    if (min > 1) {
      ShingleFilter shingleFilter = new ShingleFilter(featurePos, min, max);
      shingleFilter.setTokenSeparator(" ");
      shingleFilter.setOutputUnigrams(false);
      shingleFilter.setOutputUnigramsIfNoShingles(false);
      filter = new MinHashFilter(shingleFilter, hashCount, bucketCount, hashSetSize, bucketCount > 1);
    } else {
      filter = new MinHashFilter(featurePos, hashCount, bucketCount, hashSetSize, bucketCount > 1);
    }
    return new TokenStreamComponents(source, new RemoveDuplicatesTokenFilter(filter));
  }

}

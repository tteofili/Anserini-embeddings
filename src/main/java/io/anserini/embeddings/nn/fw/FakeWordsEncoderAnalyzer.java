package io.anserini.embeddings.nn.fw;

import io.anserini.embeddings.nn.FeatureVectorsTokenizer;
import org.apache.lucene.analysis.*;

public class FakeWordsEncoderAnalyzer extends Analyzer {

    static final String REMOVE_IT = "_";
    private final int q;

    private CharArraySet set = new CharArraySet(1, false);

    FakeWordsEncoderAnalyzer(int q) {
        this.q = q;
        this.set.add(REMOVE_IT);
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = new FeatureVectorsTokenizer();
        TokenFilter filter = new EncodeAndQuantizeFilter(t, q);
        filter = new StopFilter(filter, set);
        return new TokenStreamComponents(t, filter);
    }
}

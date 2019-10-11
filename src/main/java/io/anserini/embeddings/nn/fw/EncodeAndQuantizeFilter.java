package io.anserini.embeddings.nn.fw;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import static io.anserini.embeddings.nn.fw.FakeWordsEncoderAnalyzer.REMOVE_IT;

public final class EncodeAndQuantizeFilter extends TokenFilter {

    private static final String PREFIX = "f";
    private final CharTermAttribute termAttribute = addAttribute(CharTermAttribute.class);
    private final int q;
    private final List<String> fs = new LinkedList<>();
    private int tokenCount = 0;

    EncodeAndQuantizeFilter(TokenStream input, int q) {
        super(input);
        this.q = q;
    }

    @Override
    public boolean incrementToken() throws IOException {
        if (!fs.isEmpty()) {
            termAttribute.setEmpty();
            termAttribute.append(fs.remove(0));
            return true;
        }
        if (input.incrementToken()) {
            tokenCount++;
            String token = new String(termAttribute.buffer(), 0, termAttribute.length());
            int qv = (int) (Double.parseDouble(token) * q);
            String fw = PREFIX + tokenCount;
            for (int i = 0; i < qv - 1; i++) {
                fs.add(fw);
            }
            termAttribute.setEmpty();
            if (qv > 0) {
                termAttribute.append(fw);
            } else {
                termAttribute.append(REMOVE_IT);
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public void reset() throws IOException {
        super.reset();
        this.fs.clear();
        this.tokenCount = 0;
    }
}

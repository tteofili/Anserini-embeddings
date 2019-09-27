package io.anserini.embeddings.nn.vnlsh;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.IOException;

/**
 * {@link TokenFilter} which prepends the token / feature position plus underscore to the token itself
 * (possibly with some offset)
 */
final class FeaturePositionTokenFilter extends TokenFilter {

    private final CharTermAttribute termAttribute = addAttribute(CharTermAttribute.class);
    private final int start;
    private int tokenCount = 0;

    FeaturePositionTokenFilter(TokenStream stream) {
      this(stream, 0);
    }

    FeaturePositionTokenFilter(TokenStream stream, int start) {
        super(stream);
        this.start = start;
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (input.incrementToken()) {
        tokenCount++;
        String token = new String(termAttribute.buffer(), 0, termAttribute.length());
        termAttribute.setEmpty();
        termAttribute.append(String.valueOf(tokenCount));
        if (start > 0 && start < token.length()) {
            if (token.startsWith("-")) {
              termAttribute.append("-");
              termAttribute.append(token.substring(start + 1));
            } else {
              termAttribute.append(token.substring(start));
            }
        } else {
            termAttribute.append(token);
        }
        return true;
      } else {
        return false;
      }
    }

  @Override
  public void reset() throws IOException {
    super.reset();
    tokenCount = 0;
  }

}
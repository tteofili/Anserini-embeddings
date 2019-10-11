package io.anserini.embeddings.nn;

import io.anserini.embeddings.nn.vnlsh.VectorNgramLSHAnalyzer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.CommonTermsQuery;
import org.apache.lucene.search.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import static org.apache.lucene.search.BooleanClause.Occur.SHOULD;

/**
 * Utility methods for indexing and searching by similar vectors
 */
public class QueryUtils {

  public static Collection<String> getTokens(Analyzer analyzer, String field, String sampleTextString) throws IOException {
    Collection<String> tokens = new LinkedList<>();
    TokenStream ts = analyzer.tokenStream(field, sampleTextString);
    ts.reset();
    ts.addAttribute(CharTermAttribute.class);
    while (ts.incrementToken()) {
      CharTermAttribute charTermAttribute = ts.getAttribute(CharTermAttribute.class);
      String token = new String(charTermAttribute.buffer(), 0, charTermAttribute.length());
      tokens.add(token);
    }
    ts.end();
    ts.close();
    return tokens;
  }

  public static Query getSimQuery(Analyzer analyzer, String fieldName, String text, float similarity, float expectedTruePositive) throws IOException {
    return createFingerPrintQuery(fieldName, getTokens(analyzer, fieldName, text), similarity, expectedTruePositive);
  }

  public static Query getSimilarityQuery(IndexReader reader, int docId, String similarityField, float tp) {
    try {
      BooleanQuery.Builder similarityQuery = new BooleanQuery.Builder();
      VectorNgramLSHAnalyzer analyzer = new VectorNgramLSHAnalyzer();
      Document doc = reader.document(docId);
      String fvString = doc.get(similarityField);
      if (fvString != null && fvString.trim().length() > 0) {
        Query simQuery = QueryUtils.getSimQuery(analyzer, similarityField, fvString, 1f, tp);
        similarityQuery.add(new BooleanClause(simQuery, SHOULD));
      }
      return similarityQuery.build();
    } catch (Exception e) {
      throw new RuntimeException("could not handle similarity query for doc " + docId);
    }
  }

  private static double[] toDoubleArray(byte[] array) {
    int blockSize = Double.SIZE / Byte.SIZE;
    ByteBuffer wrap = ByteBuffer.wrap(array);
    int capacity = array.length / blockSize;
    double[] doubles = new double[capacity];
    for (int i = 0; i < capacity; i++) {
      double e = wrap.getDouble(i * blockSize);
      doubles[i] = e;
    }
    return doubles;
  }

  public static void kNNRerank(int k, boolean includeOriginalScore, double farthestDistance, List<String> fields,
                               TopDocs docs, IndexSearcher indexSearcher) throws IOException {
    ScoreDoc inputDoc = docs.scoreDocs[0]; // we assume the input doc is the first one returned
    List<Integer> toDiscard = new LinkedList<>();
    for (String fieldName : fields) {
      String value = indexSearcher.doc(inputDoc.doc).get(fieldName);
      double[] inputVector = toDoubleArray(value);
      for (int j = 0; j < docs.scoreDocs.length; j++) {
        String text = indexSearcher.doc(docs.scoreDocs[j].doc).get(fieldName);
        double[] currentVector = toDoubleArray(text);
        double distance = dist(inputVector, currentVector) + 1e-10; // constant term to avoid division by zero
        // a threshold distance beyond which current vector is discarded
        if (distance > farthestDistance || Double.isNaN(distance) || Double.isInfinite(distance)) {
          toDiscard.add(docs.scoreDocs[j].doc);
        }
        if (includeOriginalScore) {
            docs.scoreDocs[j].score += 1f / distance; // additive similarity boosting
        } else {
            docs.scoreDocs[j].score = (float) (1f / distance); // replace score
        }
      }
    }
    if (!toDiscard.isEmpty()) {
      // remove docs that are not close enough
      docs.scoreDocs = Arrays.stream(docs.scoreDocs).filter(e -> !toDiscard.contains(e.doc)).toArray(ScoreDoc[]::new);
    }
    Arrays.parallelSort(docs.scoreDocs, 0, docs.scoreDocs.length, (o1, o2) -> { // rerank scoreDocs
      return -1 * Double.compare(o1.score, o2.score);
    });
    if (docs.scoreDocs.length > k) {
      docs.scoreDocs = Arrays.copyOfRange(docs.scoreDocs, 0, k); // retain only the top k nearest neighbours
    }
  }

    private static double[] toDoubleArray(String value) {
        String[] values = value.split(" ");
        double[] result = new double[values.length];
        int i = 0;
        for (String v : values) {
            result[i] = Double.parseDouble(v);
            i++;
        }
        return result;
    }

    private static double dist(double[] x, double[] y) { // euclidean distance
    double d = 0;
    for (int i = 0; i < x.length; i++) {
      d += Math.pow(y[i] - x[i], 2);
    }
    return Math.sqrt(d);
  }

    private static Query createFingerPrintQuery(String field, Collection<String> minhashes, float similarity, float expectedTruePositive) {
        int bandSize = 1;
        if (expectedTruePositive < 1) {
            bandSize = computeBandSize(minhashes.size(), similarity, expectedTruePositive);
        }

        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        BooleanQuery.Builder childBuilder = new BooleanQuery.Builder();
        int rowInBand = 0;
        for (String minHash : minhashes) {
            TermQuery tq = new TermQuery(new Term(field, minHash));
            if (bandSize == 1) {
                builder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.SHOULD);
            } else {
                childBuilder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.MUST);
                rowInBand++;
                if (rowInBand == bandSize) {
                    builder.add(new ConstantScoreQuery(childBuilder.build()),
                            BooleanClause.Occur.SHOULD);
                    childBuilder = new BooleanQuery.Builder();
                    rowInBand = 0;
                }
            }
        }
        if (childBuilder.build().clauses().size() > 0) {
            for (String token : minhashes) {
                TermQuery tq = new TermQuery(new Term(field, token.toString()));
                childBuilder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.MUST);
                rowInBand++;
                if (rowInBand == bandSize) {
                    builder.add(new ConstantScoreQuery(childBuilder.build()),
                            BooleanClause.Occur.SHOULD);
                    break;
                }
            }
        }

        if (expectedTruePositive >= 1.0 && similarity < 1) {
            builder.setMinimumNumberShouldMatch((int) (Math.ceil(minhashes.size() * similarity)));
        }
        return builder.build();

    }

    static int computeBandSize(int numHash, double similarity, double expectedTruePositive) {
        for (int bands = 1; bands <= numHash; bands++) {
            int rowsInBand = numHash / bands;
            double truePositive = 1 - Math.pow(1 - Math.pow(similarity, rowsInBand), bands);
            if (truePositive > expectedTruePositive) {
                return rowsInBand;
            }
        }
        return 1;
    }

    public static byte[] toByteArray(List<Double> values) {
        int blockSize = Double.SIZE / Byte.SIZE;
        byte[] bytes = new byte[values.size() * blockSize];
        for (int i = 0, j = 0; i < values.size(); i++, j += blockSize) {
            ByteBuffer.wrap(bytes, j, blockSize).putDouble(values.get(i));
        }
        return bytes;
    }

    public static byte[] toByteArray(String value) {
        List<Double> doubles = new LinkedList<>();
        for (String dv : value.split(",")) {
            doubles.add(Double.parseDouble(dv));
        }
        return toByteArray(doubles);
    }

    public static String toDoubleString(byte[] bytes) {
        double[] a = toDoubleArray(bytes);
        StringBuilder builder = new StringBuilder();
        for (Double d : a) {
            if (builder.length() > 0) {
                builder.append(' ');
            }
            builder.append(d);
        }
        return builder.toString();
    }

    public static Query getCTSimQuery(Analyzer analyzer, String fieldVector, String vectorString, float mtf) throws IOException {
        CommonTermsQuery commonTermsQuery = new CommonTermsQuery(SHOULD, SHOULD, mtf);
        for (String token : getTokens(analyzer, fieldVector, vectorString)) {
            commonTermsQuery.add(new Term(fieldVector, token));
        }
        return commonTermsQuery;
    }

    public static Query getBooleanQuery(Analyzer analyzer, String fieldVector, String vectorString) throws IOException {
        BooleanQuery.Builder query = new BooleanQuery.Builder();
        for (String token : getTokens(analyzer, fieldVector, vectorString)) {
            query.add(new TermQuery(new Term(fieldVector, token)), SHOULD);
        }
        return query.build();
    }
}

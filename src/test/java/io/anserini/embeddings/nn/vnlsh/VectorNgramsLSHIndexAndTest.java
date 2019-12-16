package io.anserini.embeddings.nn.vnlsh;

import com.google.common.collect.Sets;
import io.anserini.embeddings.nn.QueryUtils;
import io.anserini.search.topicreader.TrecTopicReader;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@RunWith(Parameterized.class)
public class VectorNgramsLSHIndexAndTest {

    private static final String FIELD_WORD = "word";
    private static final String FIELD_VECTOR = "vector";
    private static final String TABLE_ROW_END = "\\\\";
    private static final String TABLE_COLUMN_SEPARATOR = " & ";
    private static final int TOP_N = 10;

    private static final int[] topKs = new int[]{10, 20, 50, 100};

    private final Logger LOG = LoggerFactory.getLogger(getClass());

    private final int decimals;
    private final int ngramSize;
    private final float similarity;
    private final float expectedTruePositive;
    private final boolean rerank;
    private final double numSamples;
    private final int hashCount;
    private final int hashSetSize;
    private final int bucketCount;

    public VectorNgramsLSHIndexAndTest(int decimals, int ngramSize, float similarity, float expectedTruePositive, boolean rerank,
                                       double numSamples, int hashCount, int hashSetSize, int bucketCount) {
        this.decimals = decimals;
        this.ngramSize = ngramSize;
        this.similarity = similarity;
        this.expectedTruePositive = expectedTruePositive;
        this.rerank = rerank;
        this.numSamples = numSamples;
        this.hashCount = hashCount;
        this.hashSetSize = hashSetSize;
        this.bucketCount = bucketCount;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE);
        // decimals, ngrams, similarity, expectedTruePositive, rerank, numSamples, hashCount, hashSetSize, bucketCount, topk
        return Arrays.asList(new Object[][]{
                {1, 2, 1f, 1f, false, 10, 1, 1, 300},
//                {1, 2, 1f, 1f, false, 20, 1, 2, 300, 50},
//                {1, 2, 1f, 1f, false, 20, 1, 3, 300, 50},
//                {1, 2, 1f, 1f, false, 20, 2, 2, 300, 50},
//                {1, 2, 1f, 1f, false, 20, 2, 3, 300, 50},
//                {1, 2, 1f, 1f, false, 10, 30, 1, 50, 50},

//                {1, 2, 1f, 0.05f, false, 20, 1, 1, 300, 50},
//                {1, 2, 0.05f, 1f, false, 20, 1, 1, 300, 50},
        });
    }

    @Test
    public void testIndexAndSearch() throws Exception {

        Path path = Paths.get("/Users/teofili/Desktop/tests/ootb-models");
        DirectoryStream<Path> stream = Files.newDirectoryStream(path);
        for (Path model : stream) {

            if (Files.isHidden(model)) {
                continue;
            }
            WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(model.toFile());

            LOG.info("model: {}", model);
            LOG.info("config: {}", this.toString());

            Path indexDir = Files.createTempDirectory("vnlsh-test");
            Directory d = FSDirectory.open(indexDir);
            Map<String, Analyzer> map = new HashMap<>();
            VectorNgramLSHAnalyzer vectorNgramLshAnalyzer = new VectorNgramLSHAnalyzer(decimals, ngramSize, hashCount, bucketCount, hashSetSize);
            map.put(FIELD_VECTOR, vectorNgramLshAnalyzer);
            Analyzer analyzer = new PerFieldAnalyzerWrapper(new StandardAnalyzer(), map);

            IndexWriter indexWriter = new IndexWriter(d, new IndexWriterConfig(analyzer));
            final AtomicInteger cnt = new AtomicInteger();

            VocabCache vocab = wordVectors.vocab();
            vocab.words().forEach(obj -> {
                String word = (String) obj;
                Document doc = new Document();

                doc.add(new StringField(FIELD_WORD, word, Field.Store.YES));
                double[] vector = wordVectors.getWordVectorMatrixNormalized(word).toDoubleVector();
                StringBuilder sb = new StringBuilder();
                for (double fv : vector) {
                    if (sb.length() > 0) {
                        sb.append(' ');
                    }
                    sb.append(fv);
                }
                doc.add(new TextField(FIELD_VECTOR, sb.toString(), rerank ? Field.Store.YES : Field.Store.NO));
                try {
                    indexWriter.addDocument(doc);
                    int cur = cnt.incrementAndGet();
                    if (cur % 100000 == 0) {
                        LOG.info("{} words added", cnt);
                    }
                } catch (IOException e) {
                    LOG.error("Error while indexing: {}", e.getLocalizedMessage());
                }
            });

            indexWriter.commit();
            indexWriter.forceMerge(4);
            LOG.info("{} words indexed", cnt.get());

            DirectoryReader reader = DirectoryReader.open(indexWriter);
            IndexSearcher searcher = new IndexSearcher(reader);

            StandardAnalyzer standardAnalyzer = new StandardAnalyzer();
            for (int topK : topKs) {
                double recall = 0;
                double time = 0d;

                TrecTopicReader trecTopicReader = new TrecTopicReader(Paths.get("src/test/resources/topics.robust04.301-450.601-700.txt"));
                SortedMap<Integer, Map<String, String>> read = trecTopicReader.read();
                int queryCount = 0;
                Collection<Map<String, String>> values = read.values();
//            for (float tp = 0.1f; tp <= 1; tp += 0.1f) {
//                for (float sim = 0.1f; sim <= 1; sim += 0.1f) {
                LOG.info("testing with {} topics", values.size());
                for (Map<String, String> topic : values) {
                    for (String word : QueryUtils.getTokens(standardAnalyzer, null, topic.get("title"))) {
                        Set<String> truth = new HashSet<>(wordVectors.wordsNearest(word, TOP_N));
                        if (!truth.isEmpty() && wordVectors.hasWord(word)) {
                            try {
                                double[] vector = wordVectors.getWordVectorMatrixNormalized(word).toDoubleVector();
                                StringBuilder sb = new StringBuilder();
                                for (double fv : vector) {
                                    if (sb.length() > 0) {
                                        sb.append(' ');
                                    }
                                    sb.append(fv);
                                }
//                                    Query simQuery = io.anserini.embeddings.nn.QueryUtils.getSimQuery(vectorNgramLshAnalyzer, FIELD_VECTOR, sb.toString(), similarity, expectedTruePositive);
//                            Query simQuery = QueryUtils.getBooleanQuery(vectorNgramLshAnalyzer, FIELD_VECTOR, sb.toString());
                                Query simQuery = io.anserini.embeddings.nn.QueryUtils.getCTSimQuery(vectorNgramLshAnalyzer, FIELD_VECTOR, sb.toString(), 0.96f);
                                TopDocs topDocs;
                                long start = System.currentTimeMillis();
                                if (rerank) {
                                    topDocs = searcher.search(simQuery, 2 * topK);
                                    if (topDocs.totalHits > 0) {
                                        QueryUtils.kNNRerank(TOP_N, false, 100d,
                                                Collections.singletonList(FIELD_VECTOR), topDocs, searcher);
                                    }
                                } else {
                                    topDocs = searcher.search(simQuery, 1 + topK);
                                }
                                time += System.currentTimeMillis() - start;
                                Set<String> observations = new HashSet<>();
                                for (ScoreDoc sd : topDocs.scoreDocs) {
                                    Document document = reader.document(sd.doc);
                                    String wordValue = document.get(FIELD_WORD);
                                    if (word.equals(wordValue)) {
                                        continue;
                                    }
                                    observations.add(wordValue);
                                }
                                double intersection = Sets.intersection(truth, observations).size();
                                double localRecall = intersection / (double) truth.size();
                                recall += localRecall;
                                queryCount++;
                            } catch (IOException e) {
                                LOG.error("search for word {} failed ({})", word, e);
                            }
                        }
                    }
                    if (numSamples > 0 && queryCount > numSamples) {
                        break;
                    }

                }
                recall /= queryCount;
                time /= queryCount;

//                    LOG.info("TP:{}, SIM:{}", tp, sim);
                LOG.info("R@{}: {}", topK, recall);
                LOG.info("avg query time: {}ms", time);
//                }
//            }
            }
            long space = FileUtils.sizeOfDirectory(indexDir.toFile()) / (1024L * 1024L);
            LOG.info("index size: {}MB", space);

            reader.close();
            indexWriter.close();
            d.close();

            FileUtils.deleteDirectory(indexDir.toFile());
        }
    }

    @Override
    public String toString() {
        return "{" +
                "decimals=" + decimals +
                ", ngramSize=" + ngramSize +
                ", similarity=" + similarity +
                ", expectedTruePositive=" + expectedTruePositive +
                ", rerank=" + rerank +
                ", hashCount=" + hashCount +
                ", bucketCount=" + bucketCount +
                ", hashSetSize=" + hashSetSize +
                '}';
    }
}

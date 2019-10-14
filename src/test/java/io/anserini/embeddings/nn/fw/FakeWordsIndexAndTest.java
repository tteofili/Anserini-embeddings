package io.anserini.embeddings.nn.fw;

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
import org.apache.lucene.search.similarities.ClassicSimilarity;
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
public class FakeWordsIndexAndTest {

    private static final String FIELD_WORD = "word";
    private static final String FIELD_VECTOR = "vector";
    private static final String TABLE_ROW_END = "\\\\";
    private static final String TABLE_COLUMN_SEPARATOR = " & ";
    private static final int TOP_N = 10;
    private static final int TOP_K = 50;

    private final Logger LOG = LoggerFactory.getLogger(getClass());

    private final int numSamples;
    private final int q;
    private final boolean rerank;

    public FakeWordsIndexAndTest(int numSamples, int q, boolean rerank) {
        this.numSamples = numSamples;
        this.q = q;
        this.rerank = rerank;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE);
        // numSamples
        return Arrays.asList(new Object[][]{
                {10, 20, false},
        });
    }

    @Test
    public void testIndexAndSearch() throws Exception {
        StringBuilder latexTableBuilder = new StringBuilder();
        latexTableBuilder
                .append("model").append(TABLE_COLUMN_SEPARATOR)
                .append("numSamples").append(TABLE_COLUMN_SEPARATOR)
                .append("q").append(TABLE_COLUMN_SEPARATOR)
                .append("recall").append(TABLE_COLUMN_SEPARATOR)
                .append("avg qtime (ms)").append(TABLE_COLUMN_SEPARATOR)
                .append("index size (MB)")
                .append(TABLE_ROW_END)
                .append("\n");

        Path path = Paths.get("/Users/teofili/Desktop/tests/ootb-models");
        DirectoryStream<Path> stream = Files.newDirectoryStream(path);
        for (Path model : stream) {

            if (Files.isHidden(model)) {
                continue;
            }
            WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(model.toFile());

            LOG.info("model: {}", model);
            LOG.info("config: {}", this.toString());

            Path indexDir = Files.createTempDirectory("fw-test");
            Directory d = FSDirectory.open(indexDir);
            Map<String, Analyzer> map = new HashMap<>();
            Analyzer fwa = new FakeWordsEncoderAnalyzer(q);
            map.put(FIELD_VECTOR, fwa);
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
            LOG.info("{} words indexed", cnt.get());

            DirectoryReader reader = DirectoryReader.open(indexWriter);
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new ClassicSimilarity());
            StandardAnalyzer standardAnalyzer = new StandardAnalyzer();
            double recall = 0;
            double time = 0d;

            TrecTopicReader trecTopicReader = new TrecTopicReader(Paths.get("src/test/resources/topics.robust04.301-450.601-700.txt"));
            SortedMap<Integer, Map<String, String>> read = trecTopicReader.read();
            int queryCount = 0;
            Collection<Map<String, String>> values = read.values();
            LOG.info("testing with {} topics", values.size());
            for (Map<String, String> topic : values) {
                for (String word : QueryUtils.getTokens(standardAnalyzer,null,  topic.get("title"))) {
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
                            Query simQuery = io.anserini.embeddings.nn.QueryUtils.getCTSimQuery(fwa, FIELD_VECTOR, sb.toString(), 0.97f);
//                            Query simQuery = io.anserini.embeddings.nn.QueryUtils.getBooleanQuery(fwa, FIELD_VECTOR, sb.toString());
                            TopDocs topDocs;
                            long start = System.currentTimeMillis();
                            if (rerank) {
                                topDocs = searcher.search(simQuery, 2 * TOP_N);
                                if (topDocs.totalHits > 0) {
                                    QueryUtils.kNNRerank(1 + TOP_N, false, 100d,
                                            Collections.singletonList(FIELD_VECTOR), topDocs, searcher);
                                }
                            } else {
                                topDocs = searcher.search(simQuery, TOP_K);
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
                if (queryCount > numSamples) {
                    break;
                }

            }
            recall /= queryCount;
            time /= queryCount;
            long space = FileUtils.sizeOfDirectory(indexDir.toFile()) / (1024L * 1024L);

            LOG.info("R@{}: {}", TOP_K, recall);
            LOG.info("avg query time: {}ms", time);
            LOG.info("index size: {}MB", space);

            latexTableBuilder
                    .append(model).append(TABLE_COLUMN_SEPARATOR)
                    .append(this.numSamples).append(TABLE_COLUMN_SEPARATOR)
                    .append(this.q).append(TABLE_COLUMN_SEPARATOR)
                    .append(this.rerank).append(TABLE_COLUMN_SEPARATOR)
                    .append(recall).append('@').append(TOP_K).append(TABLE_COLUMN_SEPARATOR)
                    .append(time).append(TABLE_COLUMN_SEPARATOR)
                    .append(space).append(TABLE_ROW_END)
                    .append("\n");
            reader.close();
            indexWriter.close();
            d.close();

            FileUtils.deleteDirectory(indexDir.toFile());
        }
        IOUtils.write(latexTableBuilder.toString(), new FileOutputStream("target/fw-" + System.currentTimeMillis() + ".txt"), Charset.defaultCharset());
    }


    @Override
    public String toString() {
        return "FakeWordsIndexAndTest{" +
                "numSamples=" + numSamples +
                ", q=" + q +
                ", rerank=" + rerank +
                '}';
    }
}

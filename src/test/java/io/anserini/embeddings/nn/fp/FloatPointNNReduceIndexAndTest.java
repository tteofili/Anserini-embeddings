package io.anserini.embeddings.nn.fp;

import com.google.common.collect.Sets;
import io.anserini.embeddings.nn.QueryUtils;
import io.anserini.search.topicreader.TrecTopicReader;
import io.anserini.util.AnalyzerUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@RunWith(Parameterized.class)
public class FloatPointNNReduceIndexAndTest {

    private static final String FIELD_WORD = "word";
    private static final String FIELD_VECTOR = "vector";
    private static final String TABLE_ROW_END = "\\\\";
    private static final String TABLE_COLUMN_SEPARATOR = " & ";
    private static final int TOP_N = 10;
    private static final int TOP_K = 10;
    private static final String FLOAT_POINT = "float_point";

    private final Logger LOG = LoggerFactory.getLogger(getClass());

    private final boolean rerank;
    private final double numSamples;

    public FloatPointNNReduceIndexAndTest(boolean rerank, double numSamples) {
        this.rerank = rerank;
        this.numSamples = numSamples;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE);
        // rerank, numSamples
        return Arrays.asList(new Object[][]{
                {false, 50},
        });
    }

    @Test
    public void testIndexAndSearch() throws Exception {
        StringBuilder latexTableBuilder = new StringBuilder();
        latexTableBuilder
                .append("model").append(TABLE_COLUMN_SEPARATOR)
                .append("numSamples").append(TABLE_COLUMN_SEPARATOR)
                .append("rerank").append(TABLE_COLUMN_SEPARATOR)
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

            LOG.info("reducing vectors");
            int dim = 8;

            INDArray reduced = PCA.pca(wordVectors.lookupTable().getWeights(), dim, false);

            LOG.info("reducing done");

            LOG.info("model: {}", model);

            Path indexDir = Files.createTempDirectory("fpnn-test");
            Directory d = FSDirectory.open(indexDir);
            Map<String, Analyzer> map = new HashMap<>();
            Analyzer analyzer = new PerFieldAnalyzerWrapper(new StandardAnalyzer(), map);

            IndexWriter indexWriter = new IndexWriter(d, new IndexWriterConfig(analyzer));
            for (int i = 0; i < reduced.rows(); i++) {
                String word = wordVectors.vocab().wordAtIndex(i);
                Document doc = new Document();
                doc.add(new StringField(FIELD_WORD, word, Field.Store.YES));
                INDArray row = reduced.getRow(i);
                float[] reducedPoint = row.toFloatVector();

                ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
                try {
                    DataOutputStream dataOut = new DataOutputStream(bytesOut);
                    double[] vector = reduced.getRow(i).toDoubleVector();
                    dataOut.writeInt(vector.length);
                    for (double v : vector) {
                        dataOut.writeFloat((float) v);
                    }
                    dataOut.close();
                } catch (IOException e) {
                    LOG.error("error {}", e.getLocalizedMessage());
                }

                doc.add(new FloatPoint(FLOAT_POINT, reducedPoint));
                doc.add(new StoredField(FIELD_VECTOR, bytesOut.toByteArray()));

                try {
                    indexWriter.addDocument(doc);
                    if (i % 100000 == 0) {
                        LOG.info("{} words added", i);
                    }
                } catch (IOException e) {
                    LOG.error("Error while indexing: {}", e.getLocalizedMessage());
                }
            }

            indexWriter.commit();

            DirectoryReader reader = DirectoryReader.open(indexWriter);

            LOG.info("{} words indexed", reader.numDocs());

            IndexSearcher searcher = new IndexSearcher(reader);

            StandardAnalyzer standardAnalyzer = new StandardAnalyzer();
            double recall = 0;
            double time = 0d;

            TrecTopicReader trecTopicReader = new TrecTopicReader(Paths.get("src/test/resources/topics.robust04.301-450.601-700.txt"));
            SortedMap<Integer, Map<String, String>> read = trecTopicReader.read();
            int queryCount = 0;
            Collection<Map<String, String>> values = read.values();
            LOG.info("testing with {} topics", values.size());
            for (Map<String, String> topic : values) {
                for (String word : AnalyzerUtils.tokenize(standardAnalyzer, topic.get("title"))) {
                    Set<String> truth = new HashSet<>(wordVectors.wordsNearest(word, TOP_N));
                    if (wordVectors.hasWord(word)) {
                        try {
                            TopDocs wordResult = searcher.search(new TermQuery(new Term(FIELD_WORD, word)), 1);
                            ScoreDoc scoreDoc = wordResult.scoreDocs[0];

                            byte[] value = reader.document(scoreDoc.doc).getField(FIELD_VECTOR).binaryValue().bytes;
                            DataInputStream in = new DataInputStream(new ByteArrayInputStream(value));

                            int l = in.readInt();
                            float[] vector = new float[l];
                            for (int n = 0; n < vector.length; n++) {
                                vector[n] = in.readFloat();
                            }
                            TopFieldDocs topDocs;
                            long start = System.currentTimeMillis();
                            if (rerank) {
                                topDocs = FloatPointNearestNeighbor.nearest(searcher, FLOAT_POINT, 2 * TOP_K, vector);
                                if (topDocs.totalHits > 0) {
                                    QueryUtils.kNNRerank(1 + TOP_N, false, 100d, Collections.singletonList(FIELD_VECTOR), topDocs, searcher);
                                }
                            } else {
                                topDocs = FloatPointNearestNeighbor.nearest(searcher, FLOAT_POINT, TOP_K, vector);
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
            long space = FileUtils.sizeOfDirectory(indexDir.toFile()) / (1024L * 1024L);

            LOG.info("R@{}: {}", TOP_K, recall);
            LOG.info("avg query time: {}ms", time);
            LOG.info("index size: {}MB", space);

            reader.close();
            indexWriter.close();
            d.close();

            FileUtils.deleteDirectory(indexDir.toFile());
        }
        IOUtils.write(latexTableBuilder.toString(), new FileOutputStream("target/fp-" + System.currentTimeMillis() + ".txt"), Charset.defaultCharset());
    }

    private INDArray postProcess(INDArray weights, int d) {
        INDArray meanWeights = weights.subRowVector(weights.mean(0));
        INDArray pc = PCA.pca(meanWeights,d, true);
//        return meanWeights.transpose().mmul(pc).mmul(pc.transpose()).transpose();
        return meanWeights.mmul(pc.transpose()).mmul(pc);
    }
}

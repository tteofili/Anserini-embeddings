package io.anserini.embeddings.nn.fw;

import com.google.common.collect.Sets;
import io.anserini.embeddings.IndexReducedWordEmbeddings;
import io.anserini.embeddings.nn.QueryUtils;
import io.anserini.search.topicreader.TrecTopicReader;
import io.anserini.util.AnalyzerUtils;
import org.apache.commons.io.FileUtils;
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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.kohsuke.args4j.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class FakeWordsRunner {

    private static final String FIELD_WORD = "word";
    private static final String FIELD_VECTOR = "vector";
    private static final int TOP_N = 10;
    private static final int[] topKs = new int[]{10, 20, 50, 100};

    public static final class Args {
        @Option(name = "-input", metaVar = "[file]", required = true, usage = "word vectors model")
        public File input;

        @Option(name = "-q", metaVar = "[int]", required = true, usage = "quantization factor")
        public int q;
    }

    public static void main(String[] args) throws Exception {

        FakeWordsRunner.Args indexArgs = new FakeWordsRunner.Args();
        CmdLineParser parser = new CmdLineParser(indexArgs, ParserProperties.defaults().withUsageWidth(90));

        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.err.println("Example: "+ FakeWordsRunner.class.getSimpleName() +
                    parser.printExample(OptionHandlerFilter.REQUIRED));
            return;
        }

        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(indexArgs.input);

        Path indexDir = Files.createTempDirectory("fw-test");
        Directory d = FSDirectory.open(indexDir);
        Map<String, Analyzer> map = new HashMap<>();
        Analyzer fwa = new FakeWordsEncoderAnalyzer(indexArgs.q);
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
            doc.add(new TextField(FIELD_VECTOR, sb.toString(), Field.Store.NO));
            try {
                indexWriter.addDocument(doc);
                int cur = cnt.incrementAndGet();
                if (cur % 100000 == 0) {
                    System.out.println(cnt + " words added");
                }
            } catch (IOException e) {
                System.err.println("Error while indexing: " + e.getLocalizedMessage());
            }
        });

        indexWriter.commit();
        System.out.println(cnt + " words indexed");
        long space = FileUtils.sizeOfDirectory(indexDir.toFile()) / (1024L * 1024L);
        System.out.println("index size: " + space + "MB");

        DirectoryReader reader = DirectoryReader.open(indexWriter);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new ClassicSimilarity());
        StandardAnalyzer standardAnalyzer = new StandardAnalyzer();

        for (int topK : topKs) {
            double recall = 0;
            double time = 0d;

            TrecTopicReader trecTopicReader = new TrecTopicReader(Paths.get("src/test/resources/topics.robust04.301-450.601-700.txt"));
            SortedMap<Integer, Map<String, String>> read = trecTopicReader.read();
            int queryCount = 0;
            Collection<Map<String, String>> values = read.values();
            for (Map<String, String> topic : values) {
                for (String word : AnalyzerUtils.tokenize(standardAnalyzer, topic.get("title"))) {
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
                            Query simQuery = io.anserini.embeddings.nn.QueryUtils.getCTSimQuery(fwa, FIELD_VECTOR,
                                    sb.toString(), 0.999f);
                            TopDocs topDocs = searcher.search(simQuery, topK);
                            long start = System.currentTimeMillis();

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
                            System.err.println("search for word " + word + "failed " + e.getLocalizedMessage());
                        }
                    }
                }
            }
            recall /= queryCount;
            time /= queryCount;

            System.out.println("R@" + topK + ": " + recall);
            System.out.println("avg query time: " + time + "ms");
        }
        reader.close();
        indexWriter.close();
        d.close();

        FileUtils.deleteDirectory(indexDir.toFile());
    }
}

/**
 * Anserini: A toolkit for reproducible information retrieval research built on Lucene
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.embeddings;

import io.anserini.util.AnalyzerUtils;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.*;

import java.io.*;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Takes an existing input index containing text documents and a word embeddings index and generates
 * a new index for document embeddings lookup (and nearest neighbour search), by averaging word embeddings of each term
 * in a document, as one of the approaches outlined in the paper "Document Embedding with Paragraph Vectors" by Andrew
 * M. Dai, Christopher Olah, Quoc V.
 *
 */
public class IndexReducedAveragedWordEmbeddings {
  private static final Logger LOG = LogManager.getLogger(IndexReducedAveragedWordEmbeddings.class);

  public static final class Args {
    @Option(name = "-input", metaVar = "[path]", required = true, usage = "input documents index path")
    public Path docs;

    @Option(name = "-words", metaVar = "[path]", required = true, usage = "input word embeddings index path")
    public Path words;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "output document embeddings index path")
    public Path output;

    @Option(name = "-dimensions", metaVar = "[int]", required = true, usage = "dimensions")
    public int dimensions;

    @Option(name = "-contentField", metaVar = "[word]", required = true, usage = "content field name")
    public String contentField;
  }

  public static final String FIELD_DOCID = "doc_id";
  public static final String FIELD_AWE_VECTOR = "vector";
  public static final String FIELD_AWE_POINT = "point";

  public static void main(String[] args) throws Exception {
    Args indexArgs = new Args();
    CmdLineParser parser = new CmdLineParser(indexArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ IndexReducedAveragedWordEmbeddings.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    final long start = System.nanoTime();
    LOG.info("Starting indexer...");

    final Analyzer analyzer = new EnglishStemmingAnalyzer("porter"); // Default used in indexing.
    final IndexWriterConfig config = new IndexWriterConfig(analyzer);
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

    final Directory words = FSDirectory.open(indexArgs.words);

    final AtomicInteger cnt = new AtomicInteger();

    final Directory inputDir = FSDirectory.open(indexArgs.docs);
    final IndexReader reader = DirectoryReader.open(inputDir);
    LOG.info("Indexing embeddings for {} documents", reader.maxDoc());

    DirectoryReader wordsReader = DirectoryReader.open(words);
    LOG.info("Using {} word embeddings", wordsReader.maxDoc());
    IndexSearcher wordVectorSearcher = new IndexSearcher(wordsReader);

    final Directory outputDocumentsDir = FSDirectory.open(indexArgs.output);
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

    final IndexWriter documentsWriter = new IndexWriter(outputDocumentsDir, config);

    for (int i = 0; i < reader.maxDoc(); i++) {

      Document inputDoc = reader.document(i);
      if (inputDoc != null) {
        Document outputDoc = new Document();

        outputDoc.add(new TextField(FIELD_DOCID, String.valueOf(i), Field.Store.YES));

        float[] average = average(indexArgs.dimensions, getWordVectors(AnalyzerUtils.tokenize(analyzer,
                inputDoc.get(indexArgs.contentField)), wordVectorSearcher, indexArgs.dimensions));
        outputDoc.add(new FloatPoint(FIELD_AWE_POINT, average));

        ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
        try {
          DataOutputStream dataOut = new DataOutputStream(bytesOut);
          dataOut.writeInt(average.length);
          for (float v : average) {
            dataOut.writeFloat(v);
          }
          dataOut.close();
        } catch (IOException e) {
          LOG.error(e);
        }

        outputDoc.add(new StoredField(FIELD_AWE_VECTOR, bytesOut.toByteArray()));
        bytesOut.close();

        try {
          documentsWriter.addDocument(outputDoc);
          int cur = cnt.incrementAndGet();
          if (cur % 100000 == 0) {
            LOG.info(cnt + " docs added.");
          }
        } catch (IOException e) {
          LOG.error(e);
        }
      }
    }

    LOG.info(cnt.get() + " docs added.");
    int numIndexed = documentsWriter.maxDoc();

    try {
      documentsWriter.commit();
    } finally {
      try {
        wordsReader.close();
        documentsWriter.close();
      } catch (IOException e) {
        LOG.error(e);
      }
    }


    long duration = TimeUnit.MILLISECONDS.convert(System.nanoTime() - start, TimeUnit.NANOSECONDS);
    LOG.info("Total " + numIndexed + " reduced docs indexed in " +
        DurationFormatUtils.formatDuration(duration, "HH:mm:ss"));
  }

  private static float[][] getWordVectors(List<String> qtokens, IndexSearcher searcher, int dimensions) throws IOException {
    float[][] floats = new float[qtokens.size()][dimensions];
    int j = 0;
    for (String token : qtokens) {
      TermQuery query = new TermQuery(new Term(IndexReducedWordEmbeddings.FIELD_WORD, token));

      TopDocs topDocs = searcher.search(query, 1);
      for ( int i = 0; i < topDocs.scoreDocs.length; i++ ) {
        Document doc = searcher.getIndexReader().document(topDocs.scoreDocs[i].doc);
        byte[] value = doc.getField(IndexReducedWordEmbeddings.FIELD_REDUCED_VECTOR).binaryValue().bytes;
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(value));

        int cnt = in.readInt();
        float[] vector = new float[cnt];
        for (int n = 0; n < vector.length; n++) {
          vector[n] = in.readFloat();
        }

        floats[j] = vector;
      }
      j++;
    }
    return floats;
  }

  private static float[] average(int dimensions, float[]... vectors) {
    float[] average = new float[dimensions];
    for (float[] vector : vectors) {
      for (int j = 0; j < vector.length; j++) {
        average[j] += vector[j];
      }
    }
    for (int j = 0; j < average.length; j++) {
      average[j] /= (float) vectors.length;
    }
    return average;
  }

}

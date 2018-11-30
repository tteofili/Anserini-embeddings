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

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionHandlerFilter;
import org.kohsuke.args4j.ParserProperties;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Takes GloVe word embeddings and creates a Lucene index for lookup. This is treating Lucene as a simple key-value store.
 */
public class IndexGloVe {
  private static final Logger LOG = LogManager.getLogger(IndexGloVe.class);

  public static final class Args {
    @Option(name = "-input", metaVar = "[file]", required = true, usage = "GloVe data")
    public File input;

    @Option(name = "-index", metaVar = "[path]", required = true, usage = "index path")
    public Path index;
  }

  public static final String FIELD_WORD = "word";
  public static final String FIELD_VECTOR = "vector";

  public static void main(String[] args) throws Exception {
    Args indexArgs = new Args();
    CmdLineParser parser = new CmdLineParser(indexArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ IndexGloVe.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    long startTime = System.currentTimeMillis();
    LOG.info("Loading GloVe vectors...");
    WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(indexArgs.input);
    LOG.info("Completed in " + (System.currentTimeMillis()-startTime)/1000 + "s elapsed.");

    final long start = System.nanoTime();
    LOG.info("Starting indexer...");

    final Directory dir = FSDirectory.open(indexArgs.index);
    final Analyzer analyzer = new EnglishStemmingAnalyzer("porter"); // Default used in indexing.
    final IndexWriterConfig config = new IndexWriterConfig(analyzer);
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

    final IndexWriter writer = new IndexWriter(dir, config);
    final AtomicInteger cnt = new AtomicInteger();

    VocabCache vocab = wordVectors.vocab();
    vocab.words().forEach(obj -> {
      String word = (String) obj;
      Document doc = new Document();

      double[] vector = wordVectors.getWordVector(word);
      ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
      try {
        DataOutputStream dataOut = new DataOutputStream(bytesOut);
        dataOut.writeInt(vector.length);
        for (int i = 0; i < vector.length; i++) {
          dataOut.writeFloat((float) vector[i]);
        }
        dataOut.close();
      } catch (IOException e) {
        LOG.error(e);
      }

      doc.add(new TextField(FIELD_WORD, word, Field.Store.YES));
      doc.add(new StoredField(FIELD_VECTOR, bytesOut.toByteArray()));
      try {
        writer.addDocument(doc);
        int cur = cnt.incrementAndGet();
        if (cur % 100000 == 0) {
          LOG.info(cnt + " words added.");
        }
      } catch (IOException e) {
        LOG.error(e);
      }
    });

    LOG.info(cnt.get() + " words added.");
    int numIndexed = writer.maxDoc();

    try {
      writer.commit();
    } finally {
      try {
        writer.close();
      } catch (IOException e) {
        LOG.error(e);
      }
    }

    long duration = TimeUnit.MILLISECONDS.convert(System.nanoTime() - start, TimeUnit.NANOSECONDS);
    LOG.info("Total " + numIndexed + " words indexed in " +
        DurationFormatUtils.formatDuration(duration, "HH:mm:ss"));
  }
}

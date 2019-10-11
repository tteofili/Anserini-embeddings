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

import io.anserini.embeddings.nn.QueryUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPointNearestNeighbor;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.*;

import java.nio.file.Path;
import java.util.Collection;

/**
 * Example illustrating how to perform AWE vectors (see {@code IndexReducedAveragedWordEmbeddings}) nearest neighbour
 * using Lucene {@link FloatPointNearestNeighbor}.
 */
public class AveragedWordEmbeddingsNearestNeighbour {
  public static final class Args {
    @Option(name = "-text", metaVar = "[word]", required = true, usage = "text to look up")
    public String text;

    @Option(name = "-awe-index", metaVar = "[path]", required = true, usage = "awe index path")
    public Path aweIndex;

    @Option(name = "-word-index", metaVar = "[path]", required = true, usage = "word vectors index path")
    public Path wordIndex;

    @Option(name = "-text-index", metaVar = "[path]", required = true, usage = "text index path")
    public Path index;

    @Option(name = "-contentField", metaVar = "[word]", required = true, usage = "content field name")
    public String contentField;
  }

  public static void main(String[] args) throws Exception {
    Args lookupArgs = new Args();
    CmdLineParser parser = new CmdLineParser(lookupArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ AveragedWordEmbeddingsNearestNeighbour.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }
    FSDirectory indexDir = FSDirectory.open(lookupArgs.index);
    DirectoryReader open = DirectoryReader.open(indexDir);

    FSDirectory directory = FSDirectory.open(lookupArgs.wordIndex);
    IndexReader reader = DirectoryReader.open(directory);
    IndexSearcher searcher = new IndexSearcher(reader);

    FSDirectory aweDirectory = FSDirectory.open(lookupArgs.aweIndex);
    IndexReader aweReader = DirectoryReader.open(aweDirectory);
    IndexSearcher aweSearcher = new IndexSearcher(reader);

    try {
      Analyzer analyzer = new EnglishStemmingAnalyzer("porter"); // Default used in indexing.
      Collection<String> qtokens = QueryUtils.getTokens(analyzer, null, lookupArgs.text);

      float[] average = IndexReducedAveragedWordEmbeddings.average(IndexReducedAveragedWordEmbeddings.getWordVectors(qtokens, searcher));

      TopFieldDocs nearest = FloatPointNearestNeighbor.nearest(aweSearcher, IndexReducedAveragedWordEmbeddings.FIELD_AWE_POINT, 5, average);
      for (ScoreDoc sd : nearest.scoreDocs) {
        Document document = aweReader.document(sd.doc);
        String docId = document.getField(IndexReducedAveragedWordEmbeddings.FIELD_DOCID).stringValue();

        String value = open.document(Integer.parseInt(docId)).get(lookupArgs.contentField);
        System.out.println(String.format("doc #%s : %s", docId, value));
      }
    } finally {
      open.close();
      indexDir.close();
      reader.close();
      directory.close();
      aweReader.close();
      aweDirectory.close();
    }
  }
}

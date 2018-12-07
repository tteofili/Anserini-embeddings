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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPointNearestNeighbor;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.*;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.nio.file.Path;
import java.util.List;

/**
 * Example illustrating how to perform word vectors nearest neighbour using Lucene {@link FloatPointNearestNeighbor}.
 */
public class NearestNeighbour {
  public static final class Args {
    @Option(name = "-word", metaVar = "[word]", required = true, usage = "word to look up")
    public String word;

    @Option(name = "-index", metaVar = "[path]", required = true, usage = "index path")
    public Path index;
  }

  public static void main(String[] args) throws Exception {
    Args lookupArgs = new Args();
    CmdLineParser parser = new CmdLineParser(lookupArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: "+ NearestNeighbour.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    IndexReader reader = DirectoryReader.open(FSDirectory.open(lookupArgs.index));
    IndexSearcher searcher = new IndexSearcher(reader);

    Analyzer analyzer = new EnglishStemmingAnalyzer("porter"); // Default used in indexing.
    List<String> qtokens = AnalyzerUtils.tokenize(analyzer, lookupArgs.word);
    if (qtokens.size() != 1) {
      System.err.println("Error: word tokenizes to more than one token");
      System.exit(-1);
    }

    TermQuery query = new TermQuery(new Term(IndexWordEmbeddings.FIELD_WORD, qtokens.get(0)));

    TopDocs topDocs = searcher.search(query, 1);
    if (topDocs.totalHits == 0) {
      System.err.println("Error: term not found!");
      System.exit(-1);
    }


    for ( int i=0; i<topDocs.scoreDocs.length; i++ ) {
      Document doc = reader.document(topDocs.scoreDocs[i].doc);
      byte[] value = doc.getField(IndexReducedWordEmbeddings.FIELD_REDUCED_VECTOR).binaryValue().bytes;
      DataInputStream in = new DataInputStream(new ByteArrayInputStream(value));

      int cnt = in.readInt();
      float[] vector = new float[cnt];
      for (int n=0; n<vector.length; n++) {
        vector[n] = in.readFloat();
      }

      TopFieldDocs nearest = FloatPointNearestNeighbor.nearest(searcher, IndexReducedWordEmbeddings.FIELD_POINT, 5, vector);
      for (ScoreDoc sd : nearest.scoreDocs) {
        Document document = reader.document(sd.doc);
        System.out.println(String.format("%s", document.getField(IndexWordEmbeddings.FIELD_WORD).stringValue()));
      }
    }
  }
}

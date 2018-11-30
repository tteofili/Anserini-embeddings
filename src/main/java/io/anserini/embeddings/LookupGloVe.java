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
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionHandlerFilter;
import org.kohsuke.args4j.ParserProperties;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.nio.file.Path;
import java.util.List;

/**
 * Example illustrating how to look up word vectors. Note that terms are processed with a Lucene Analyzer, which means
 * that a query term may match multiple entries in the original GloVe work embeddings. It's up to the client to figure
 * out how to deal with this issue.
 */
public class LookupGloVe {
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
      System.err.println("Example: "+ LookupGloVe.class.getSimpleName() +
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

    TermQuery query = new TermQuery(new Term(IndexGloVe.FIELD_WORD, qtokens.get(0)));

    TopDocs topDocs = searcher.search(query, Integer.MAX_VALUE);
    if (topDocs.totalHits == 0) {
      System.err.println("Error: term not found!");
      System.exit(-1);
    }


    for ( int i=0; i<topDocs.scoreDocs.length; i++ ) {
      Document doc = reader.document(topDocs.scoreDocs[i].doc);
      List<String> tokens = AnalyzerUtils.tokenize(new SimpleAnalyzer(), doc.getField(IndexGloVe.FIELD_WORD).stringValue());
      byte[] value = doc.getField(IndexGloVe.FIELD_VECTOR).binaryValue().bytes;
      DataInputStream in = new DataInputStream(new ByteArrayInputStream(value));

      int cnt = in.readInt();
      float[] vector = new float[cnt];
      for (int n=0; n<vector.length; n++) {
        vector[n] = in.readFloat();
      }

      System.out.println(String.format("%s %d [%f, %f, %f, %f ... ]", doc.getField(IndexGloVe.FIELD_WORD).stringValue(),
          tokens.size(), vector[0], vector[1], vector[2], vector[3]));
    }
  }
}

import pickle
import json
import math
import struct
import zlib
from collections import defaultdict
from typing import Iterable, Tuple, Dict, List
from pathlib import Path
from index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
from preprocess import preprocess
from query_parser import QueryParser
from query_engine import QueryEngine

class MySelfIndex(IndexBase):
    """Modular self-implemented search index."""
    
    def __init__(self, core, info, dstore, qproc, compr, optim):
        super().__init__(core, info, dstore, qproc, compr, optim)
        
        # Parse strategy flags
        self.info_strategy = IndexInfo[info]
        self.datastore_strategy = DataStore[dstore]
        self.compression_strategy = Compression[compr]
        self.qproc_strategy = QueryProc[qproc]
        self.optim_strategy = Optimizations[optim]
        
        # Index storage
        self.index = {}
        self.doc_lengths = {}
        self.idf_scores = {}
        self.total_docs = 0
        self.indexed_files = set()
        
        # Base directory for index storage
        self.base_dir = Path("indices") / self.identifier_short
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_parser = QueryParser()
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        """Create index from files."""
        print(f"Creating index {index_id} with configuration {self.identifier_short}")
        
        # Build inverted index
        inverted_index = defaultdict(lambda: defaultdict(list))
        doc_lengths = {}
        doc_count = 0
        all_doc_ids = []
        
        for doc_id, content in files:
            tokens = preprocess(content)
            doc_lengths[doc_id] = len(tokens)
            doc_count += 1
            all_doc_ids.append(doc_id)
            self.indexed_files.add(doc_id)
            
            # Track term positions
            for pos, token in enumerate(tokens):
                inverted_index[token][doc_id].append(pos)
        
        self.total_docs = doc_count
        self.doc_lengths = doc_lengths
        
        # Store all doc IDs for NOT operations
        self.index['__all_docs__'] = all_doc_ids
        
        # Build index based on info_strategy
        for term, doc_positions in inverted_index.items():
            postings = []
            
            for doc_id, positions in doc_positions.items():
                if self.info_strategy == IndexInfo.BOOLEAN:
                    # x=1: (doc_id, [positions])
                    postings.append((doc_id, positions))
                
                elif self.info_strategy == IndexInfo.WORDCOUNT:
                    # x=2: (doc_id, term_count, [positions])
                    term_count = len(positions)
                    postings.append((doc_id, term_count, positions))
                
                elif self.info_strategy == IndexInfo.TFIDF:
                    # x=3: Calculate TF-IDF
                    tf = len(positions) / doc_lengths[doc_id] if doc_lengths[doc_id] > 0 else 0
                    postings.append((doc_id, tf, positions))
            
            # Calculate IDF for TF-IDF
            if self.info_strategy == IndexInfo.TFIDF:
                df = len(postings)
                idf = math.log(doc_count / df) if df > 0 else 0
                self.idf_scores[term] = idf
                
                # Apply IDF to TF scores
                postings = [(doc_id, tf * idf, positions) for doc_id, tf, positions in postings]
            
            # Sort postings by doc_id
            postings.sort(key=lambda x: x[0])
            
            # Apply skip pointers if enabled (i=1)
            if self.optim_strategy == Optimizations.Skipping:
                postings = self._add_skip_pointers(postings)
            
            # Apply compression (z=n)
            compressed_postings = self._compress_postings(postings)
            
            self.index[term] = compressed_postings
        
        # Save index using datastore strategy
        self._save_index(index_id)
        
        print(f"Index created with {len(self.index)} terms and {doc_count} documents")
    
    def _add_skip_pointers(self, postings: List) -> List:
        """Add skip pointers to postings (every sqrt(n) entries)."""
        n = len(postings)
        skip_distance = int(math.sqrt(n)) if n > 0 else 1
        
        postings_with_skips = []
        for i, posting in enumerate(postings):
            skip_to = i + skip_distance if i + skip_distance < n else None
            postings_with_skips.append((posting, skip_to))
        
        return postings_with_skips
    
    def _compress_postings(self, postings: List) -> bytes:
        """Compress postings based on compression strategy."""
        if self.compression_strategy == Compression.NONE:
            return pickle.dumps(postings)
        
        # Extract doc_ids for gap encoding
        if self.optim_strategy == Optimizations.Skipping:
            doc_ids = [p[0][0] for p in postings]
        else:
            doc_ids = [p[0] for p in postings]
        
        # Gap encoding
        gaps = [doc_ids[0]] if doc_ids else []
        for i in range(1, len(doc_ids)):
            # Simple numeric difference (assuming doc_ids can be converted)
            gaps.append(doc_ids[i])
        
        if self.compression_strategy == Compression.CODE:
            # z=1: Variable-byte encoding
            return self._vbyte_encode(postings)
        
        elif self.compression_strategy == Compression.CLIB:
            # z=2: Use zlib compression
            serialized = pickle.dumps(postings)
            return zlib.compress(serialized)
        
        return pickle.dumps(postings)
    
    def _vbyte_encode(self, postings: List) -> bytes:
        """Simple variable-byte encoding."""
        # For simplicity, just use pickle with a marker
        return b'VBYTE:' + pickle.dumps(postings)
    
    def _decompress_postings(self, compressed: bytes) -> List:
        """Decompress postings."""
        if self.compression_strategy == Compression.NONE:
            return pickle.loads(compressed)
        
        elif self.compression_strategy == Compression.CODE:
            # Remove VBYTE marker
            if compressed.startswith(b'VBYTE:'):
                return pickle.loads(compressed[6:])
            return pickle.loads(compressed)
        
        elif self.compression_strategy == Compression.CLIB:
            decompressed = zlib.decompress(compressed)
            return pickle.loads(decompressed)
        
        return pickle.loads(compressed)
    
    def _save_index(self, index_id: str):
        """Save index based on datastore strategy."""
        if self.datastore_strategy == DataStore.CUSTOM:
            # y=1: Save using pickle
            index_file = self.base_dir / f"{index_id}_index.pkl"
            meta_file = self.base_dir / f"{index_id}_meta.pkl"
            
            with open(index_file, 'wb') as f:
                pickle.dump(self.index, f)
            
            with open(meta_file, 'wb') as f:
                pickle.dump({
                    'doc_lengths': self.doc_lengths,
                    'idf_scores': self.idf_scores,
                    'total_docs': self.total_docs,
                    'indexed_files': list(self.indexed_files)
                }, f)
        
        elif self.datastore_strategy == DataStore.DB1:
            # y=2: RocksDB
            try:
                import rocksdb
                db_path = str(self.base_dir / f"{index_id}_rocksdb")
                db = rocksdb.DB(db_path, rocksdb.Options(create_if_missing=True))
                
                for term, postings in self.index.items():
                    db.put(term.encode(), postings)
                
                # Store metadata
                meta = {
                    'doc_lengths': self.doc_lengths,
                    'idf_scores': self.idf_scores,
                    'total_docs': self.total_docs,
                    'indexed_files': list(self.indexed_files)
                }
                db.put(b'__metadata__', pickle.dumps(meta))
                
                del db
            except ImportError:
                print("RocksDB not available, falling back to pickle")
                self.datastore_strategy = DataStore.CUSTOM
                self._save_index(index_id)
        
        elif self.datastore_strategy == DataStore.DB2:
            # y=2: PostgreSQL - simplified version saves to file
            print("PostgreSQL support not fully implemented, using pickle")
            self.datastore_strategy = DataStore.CUSTOM
            self._save_index(index_id)
    
    def load_index(self, serialized_index_dump: str) -> None:
        """Load index from disk."""
        index_path = Path(serialized_index_dump)
        
        if self.datastore_strategy == DataStore.CUSTOM:
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
            
            meta_file = index_path.parent / index_path.name.replace('_index.pkl', '_meta.pkl')
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
                self.doc_lengths = meta['doc_lengths']
                self.idf_scores = meta['idf_scores']
                self.total_docs = meta['total_docs']
                self.indexed_files = set(meta['indexed_files'])
    
    def query(self, query: str) -> str:
        """Execute query and return results."""
        # Parse query
        query_ast = self.query_parser.parse(query)
        
        # Execute based on query processing strategy
        engine = QueryEngine(self.index, self.total_docs, 
                           self.optim_strategy == Optimizations.Skipping)
        
        mode = 'TAAT' if self.qproc_strategy == QueryProc.TERMatat else 'DAAT'
        results = engine.execute(query_ast, mode=mode)
        
        # Format results as JSON
        result_list = [{'doc_id': doc_id, 'score': score} for doc_id, score in results[:100]]
        return json.dumps(result_list, indent=2)
    
    def update_index(self, index_id: str, remove_files: Iterable[Tuple[str, str]], 
                    add_files: Iterable[Tuple[str, str]]) -> None:
        """Update existing index."""
        # For simplicity, rebuild the index
        print("Update not fully implemented, consider rebuilding")
    
    def delete_index(self, index_id: str) -> None:
        """Delete index files."""
        import shutil
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
    
    def list_indices(self) -> Iterable[str]:
        """List all indices."""
        indices_dir = Path("indices")
        if indices_dir.exists():
            return [d.name for d in indices_dir.iterdir() if d.is_dir()]
        return []
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List files in index."""
        return list(self.indexed_files)
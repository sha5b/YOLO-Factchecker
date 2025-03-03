import os
import json
import shutil
import logging
import numpy as np
import pickle
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib
import glob

logger = logging.getLogger("knowledge_base")

class KnowledgeBase:
    """
    Local knowledge base for storing and retrieving facts.
    Uses vector embeddings for semantic search.
    Supports multiple vector database backends (FAISS, Chroma).
    """
    
    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_db: str = "faiss",
        embedding_device: str = "auto",
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        cache_embeddings: bool = True,
        recreate: bool = False
    ):
        """
        Initialize knowledge base.
        
        Args:
            db_path: Path to knowledge base directory
            embedding_model: Name or path to sentence-transformers model
            vector_db: Vector database backend ('faiss' or 'chroma')
            embedding_device: Device for embedding model ('cpu', 'cuda', 'auto')
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            cache_embeddings: Whether to cache embeddings
            recreate: Whether to recreate the database if it exists
        """
        self.db_path = Path(db_path)
        self.vector_db = vector_db.lower()
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_embeddings = cache_embeddings
        
        # Determine device
        if embedding_device == "auto":
            import torch
            self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.embedding_device = embedding_device
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_db(recreate)
        
        logger.info(f"Knowledge base initialized at {self.db_path}")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.db_path / "sources", exist_ok=True)
        os.makedirs(self.db_path / "chunks", exist_ok=True)
        os.makedirs(self.db_path / "embeddings", exist_ok=True)
        os.makedirs(self.db_path / "vector_db", exist_ok=True)
        os.makedirs(self.db_path / "metadata", exist_ok=True)
    
    def _init_embedding_model(self) -> None:
        """Initialize embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check cache first
            cache_dir = self.db_path / "embeddings" / "model"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=str(cache_dir),
                device=self.embedding_device
            )
            
            # Get embedding dimension
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            # Save model info
            with open(self.db_path / "metadata" / "embedding_model.json", "w") as f:
                json.dump({
                    "model_name": self.embedding_model_name,
                    "dimension": self.embedding_dim,
                    "device": self.embedding_device
                }, f, indent=2)
        
        except ImportError:
            logger.error("Failed to import sentence_transformers. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _init_vector_db(self, recreate: bool = False) -> None:
        """
        Initialize vector database.
        
        Args:
            recreate: Whether to recreate the database if it exists
        """
        if self.vector_db == "faiss":
            self._init_faiss(recreate)
        elif self.vector_db == "chroma":
            self._init_chroma(recreate)
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db}")
    
    def _init_faiss(self, recreate: bool = False) -> None:
        """
        Initialize FAISS vector database.
        
        Args:
            recreate: Whether to recreate the database if it exists
        """
        try:
            import faiss
            
            db_file = self.db_path / "vector_db" / "faiss_index.bin"
            mappings_file = self.db_path / "vector_db" / "faiss_mappings.pkl"
            
            # Check if database exists
            if db_file.exists() and not recreate:
                logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(str(db_file))
                
                with open(mappings_file, "rb") as f:
                    self.id_to_data = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            else:
                logger.info("Creating new FAISS index")
                # Create a new index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.id_to_data = {}
                
                if recreate and db_file.exists():
                    logger.info("Recreating FAISS index")
                    os.remove(db_file)
                    if mappings_file.exists():
                        os.remove(mappings_file)
        
        except ImportError:
            logger.error("Failed to import faiss. Install with: pip install faiss-cpu or faiss-gpu")
            raise
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    def _init_chroma(self, recreate: bool = False) -> None:
        """
        Initialize Chroma vector database.
        
        Args:
            recreate: Whether to recreate the database if it exists
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            chroma_dir = self.db_path / "vector_db" / "chroma"
            os.makedirs(chroma_dir, exist_ok=True)
            
            # Initialize client
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Check if collection exists
            try:
                if recreate:
                    # Delete existing collection if recreate is True
                    try:
                        self.chroma_client.delete_collection("facts")
                        logger.info("Deleted existing Chroma collection")
                    except:
                        pass
                
                # Get or create collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="facts",
                    embedding_function=None,  # We'll use our own embeddings
                    metadata={"description": "Knowledge base facts"}
                )
                
                logger.info(f"Initialized Chroma collection with {self.collection.count()} documents")
            
            except Exception as e:
                logger.error(f"Error with Chroma collection: {str(e)}")
                raise
        
        except ImportError:
            logger.error("Failed to import chromadb. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Error initializing Chroma: {str(e)}")
            raise
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Document metadata
            source_id: Source identifier (generated if None)
            chunk: Whether to chunk the document
            
        Returns:
            List of chunk IDs
        """
        if metadata is None:
            metadata = {}
        
        # Generate source ID if not provided
        if source_id is None:
            source_id = self._generate_id(text)
        
        # Store source document
        source_path = self.db_path / "sources" / f"{source_id}.json"
        with open(source_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": source_id,
                "text": text,
                "metadata": metadata,
                "added_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        # Chunk document if needed
        if chunk:
            chunks = self._chunk_text(text)
            chunk_ids = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{source_id}_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "source_id": source_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks)
                })
                
                # Save chunk
                chunk_path = self.db_path / "chunks" / f"{chunk_id}.json"
                with open(chunk_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    }, f, ensure_ascii=False, indent=2)
                
                # Add to vector database
                self._add_to_vector_db(chunk_id, chunk_text, chunk_metadata)
                chunk_ids.append(chunk_id)
            
            logger.info(f"Added document {source_id} with {len(chunks)} chunks")
            return chunk_ids
        
        else:
            # Add as single document
            self._add_to_vector_db(source_id, text, metadata)
            logger.info(f"Added document {source_id} as single chunk")
            return [source_id]
    
    def _add_to_vector_db(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a document to the vector database.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Document metadata
        """
        # Generate embedding
        embedding = self._get_embedding(text)
        
        if self.vector_db == "faiss":
            # Add to FAISS
            self._add_to_faiss(doc_id, embedding, text, metadata)
        
        elif self.vector_db == "chroma":
            # Add to Chroma
            self._add_to_chroma(doc_id, embedding, text, metadata)
    
    def _add_to_faiss(
        self,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a document to FAISS.
        
        Args:
            doc_id: Document ID
            embedding: Document embedding
            text: Document text
            metadata: Document metadata
        """
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add to index
        self.index.add(embedding.reshape(1, -1))
        
        # Map ID to data
        idx = self.index.ntotal - 1
        self.id_to_data[idx] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata
        }
        
        # Save index and mappings periodically
        if self.index.ntotal % 100 == 0:
            self._save_faiss()
    
    def _add_to_chroma(
        self,
        doc_id: str,
        embedding: np.ndarray,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a document to Chroma.
        
        Args:
            doc_id: Document ID
            embedding: Document embedding
            text: Document text
            metadata: Document metadata
        """
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata]
        )
    
    def _save_faiss(self) -> None:
        """Save FAISS index and mappings."""
        if self.vector_db != "faiss":
            return
        
        # Save index
        faiss_path = self.db_path / "vector_db" / "faiss_index.bin"
        import faiss
        faiss.write_index(self.index, str(faiss_path))
        
        # Save mappings
        mappings_path = self.db_path / "vector_db" / "faiss_mappings.pkl"
        with open(mappings_path, "wb") as f:
            pickle.dump(self.id_to_data, f)
        
        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        if self.vector_db == "faiss":
            return self._search_faiss(query_embedding, top_k, threshold, filters)
        elif self.vector_db == "chroma":
            return self._search_chroma(query_embedding, top_k, threshold, filters)
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db}")
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search FAISS index.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        # Normalize embedding for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k=top_k * 2)
        
        # Process results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS padding for not enough results
                continue
            
            # Get document data
            if idx not in self.id_to_data:
                logger.warning(f"Index {idx} not found in mappings")
                continue
            
            data = self.id_to_data[idx]
            score = float(distance)  # Convert from numpy float
            
            # Apply threshold
            if score < threshold:
                continue
            
            # Apply filters
            if filters and not self._apply_filters(data["metadata"], filters):
                continue
            
            # Add to results
            results.append({
                "id": data["id"],
                "text": data["text"],
                "metadata": data["metadata"],
                "score": score
            })
            
            # Break if we have enough results after filtering
            if len(results) >= top_k:
                break
        
        return results[:top_k]
    
    def _search_chroma(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Chroma collection.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        # Prepare filters
        where_clause = {}
        if filters:
            # Convert to Chroma filter format
            for key, value in filters.items():
                where_clause[key] = value
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, text, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to score (Chroma returns L2 distance)
                score = 1.0 - (distance / 2.0)  # Normalize to 0-1 range
                
                # Apply threshold
                if score < threshold:
                    continue
                
                # Add to results
                formatted_results.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "score": score
                })
        
        return formatted_results
    
    def _apply_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Apply metadata filters.
        
        Args:
            metadata: Document metadata
            filters: Metadata filters
            
        Returns:
            Whether the document passes filters
        """
        for key, value in filters.items():
            # Skip if key not in metadata
            if key not in metadata:
                return False
            
            # Handle different filter types
            if isinstance(value, list):
                # List filter (any match)
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Dictionary filter (range)
                if "$gt" in value and metadata[key] <= value["$gt"]:
                    return False
                if "$lt" in value and metadata[key] >= value["$lt"]:
                    return False
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$ne" in value and metadata[key] == value["$ne"]:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding as numpy array
        """
        # Check cache if enabled
        if self.cache_embeddings:
            cache_key = self._generate_cache_key(text)
            cache_path = self.db_path / "embeddings" / f"{cache_key}.npy"
            
            # Return from cache if exists
            if cache_path.exists():
                return np.load(str(cache_path))
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Save to cache if enabled
        if self.cache_embeddings:
            np.save(str(cache_path), embedding)
        
        return embedding
    
    def _generate_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        # Hash text for cache key
        return hashlib.md5(text.encode()).hexdigest()
    
    def _generate_id(self, text: str) -> str:
        """
        Generate document ID.
        
        Args:
            text: Document text
            
        Returns:
            Document ID
        """
        # Generate unique ID based on text and timestamp
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()[:10]
        return f"doc_{now}_{text_hash}"
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Simple chunking based on paragraphs and size
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue
            
            # If adding paragraph exceeds chunk size, save current chunk and start new
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is longer than chunk size, split it
                if len(para) > self.chunk_size:
                    words = para.split()
                    temp_chunk = ""
                    
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > self.chunk_size:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Handle overlaps
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i > 0:
                    words = chunks[i-1].split()
                    overlap_words = words[-min(self.chunk_overlap, len(words)):]
                    overlap_text = " ".join(overlap_words)
                    
                    # Add overlap to beginning of current chunk
                    chunk = overlap_text + " " + chunk
                
                overlapped_chunks.append(chunk)
            
            return overlapped_chunks
        
        return chunks
    
    def build_from_directory(
        self,
        directory: str,
        extensions: List[str] = [".txt", ".md", ".json"],
        recursive: bool = True,
        metadata_fn: Optional[callable] = None
    ) -> int:
        """
        Build knowledge base from directory of files.
        
        Args:
            directory: Directory containing documents
            extensions: File extensions to include
            recursive: Whether to search recursively
            metadata_fn: Function to extract metadata from file path
            
        Returns:
            Number of documents added
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find files
        files = []
        for ext in extensions:
            if recursive:
                files.extend(directory.glob(f"**/*{ext}"))
            else:
                files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files in {directory}")
        
        # Process files
        count = 0
        for file_path in files:
            try:
                # Read file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Generate metadata
                if metadata_fn:
                    metadata = metadata_fn(file_path)
                else:
                    metadata = {
                        "filename": file_path.name,
                        "path": str(file_path.relative_to(directory)),
                        "extension": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                
                # Generate source ID
                source_id = f"file_{file_path.stem}"
                
                # Add to knowledge base
                self.add_document(content, metadata, source_id)
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"Processed {count}/{len(files)} files")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Save final state
        if self.vector_db == "faiss":
            self._save_faiss()
        
        logger.info(f"Added {count} documents to knowledge base")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Count files
        source_files = list(Path(self.db_path / "sources").glob("*.json"))
        chunk_files = list(Path(self.db_path / "chunks").glob("*.json"))
        
        # Get vector DB stats
        if self.vector_db == "faiss":
            vector_count = self.index.ntotal
        elif self.vector_db == "chroma":
            vector_count = self.collection.count()
        else:
            vector_count = 0
        
        return {
            "sources": len(source_files),
            "chunks": len(chunk_files),
            "vectors": vector_count,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dim,
            "vector_db": self.vector_db,
            "last_updated": datetime.now().isoformat()
        }
    
    def save_stats(self) -> None:
        """Save knowledge base statistics."""
        stats = self.get_stats()
        with open(self.db_path / "metadata" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    
    def clear(self) -> None:
        """Clear knowledge base."""
        # Clear vector DB
        if self.vector_db == "faiss":
            import faiss
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.id_to_data = {}
            self._save_faiss()
        
        elif self.vector_db == "chroma":
            try:
                self.chroma_client.delete_collection("facts")
                self.collection = self.chroma_client.create_collection(
                    name="facts",
                    embedding_function=None,
                    metadata={"description": "Knowledge base facts"}
                )
            except Exception as e:
                logger.error(f"Error clearing Chroma: {str(e)}")
        
        # Clear files
        try:
            for path in [
                self.db_path / "sources",
                self.db_path / "chunks",
                self.db_path / "embeddings"
            ]:
                # Keep directories but remove files
                for file in path.glob("*.*"):
                    if file.is_file():
                        file.unlink()
        
        except Exception as e:
            logger.error(f"Error clearing files: {str(e)}")
        
        logger.info("Knowledge base cleared")
    
    def fact_check(
        self,
        claim: str,
        top_k: int = 5,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Fact check a claim against the knowledge base.
        
        Args:
            claim: Claim to check
            top_k: Number of documents to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            Fact check result
        """
        # Search for relevant documents
        results = self.search(claim, top_k=top_k, threshold=threshold)
        
        if not results:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "reason": "No relevant information found",
                "sources": []
            }
        
        # Calculate simple confidence based on similarity scores
        avg_score = sum(r["score"] for r in results) / len(results)
        max_score = max(r["score"] for r in results)
        confidence = (avg_score + max_score) / 2  # Simple heuristic
        
        # Collect source contexts
        sources = []
        for result in results:
            sources.append({
                "id": result["id"],
                "text": result["text"],
                "score": result["score"],
                "metadata": result["metadata"]
            })
        
        return {
            "claim": claim,
            "verified": confidence > 0.7,  # Simple threshold
            "confidence": confidence,
            "reason": "Based on similarity with known facts",
            "sources": sources
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Base")
    parser.add_argument("--build", type=str, help="Build KB from directory")
    parser.add_argument("--db_path", type=str, default="data/knowledge_base", help="Knowledge base path")
    parser.add_argument("--vector_db", type=str, default="faiss", choices=["faiss", "chroma"], help="Vector DB backend")
    parser.add_argument("--clear", action="store_true", help="Clear existing KB")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--fact_check", type=str, help="Fact check claim")
    parser.add_argument("--stats", action="store_true", help="Print KB stats")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize knowledge base
    kb = KnowledgeBase(
        db_path=args.db_path,
        vector_db=args.vector_db,
        recreate=args.clear
    )
    
    if args.clear:
        kb.clear()
    
    if args.build:
        kb.build_from_directory(args.build)
    
    if args.search:
        results = kb.search(args.search, top_k=5)
        print(f"\nSearch results for: {args.search}")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['id']} (Score: {result['score']:.4f})")
            print(f"   {result['text'][:100]}...")
    
    if args.fact_check:
        result = kb.fact_check(args.fact_check)
        print(f"\nFact check: {args.fact_check}")
        print(f"Verified: {result['verified']} (Confidence: {result['confidence']:.4f})")
        print(f"Reason: {result['reason']}")
        if result['sources']:
            print(f"Top source: {result['sources'][0]['text'][:100]}...")
    
    if args.stats:
        stats = kb.get_stats()
        print("\nKnowledge Base Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

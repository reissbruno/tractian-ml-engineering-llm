"""
Service for generating embeddings using OpenAI text-embedding-3-large.
"""

import asyncio
import os
from typing import List

from langchain_openai import OpenAIEmbeddings

from src.logger import logger


class EmbeddingsService:
    """
    Embeddings service using OpenAI text-embedding-3-large.
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initializes the embeddings service.

        Args:
            model_name: OpenAI model name (default: text-embedding-3-large)
        """
        self.model_name = model_name
        self.embeddings = None
        logger.info(f"Initializing EmbeddingsService with OpenAI model: {model_name}")

    def load_model(self):
        """
        Initializes the OpenAI embeddings client (lazy loading).
        """
        if self.embeddings is None:
            logger.info(f"Initializing OpenAI Embeddings: {self.model_name}")
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not defined in environment variables")

                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    api_key=api_key
                )
                logger.info("OpenAI Embeddings initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI Embeddings: {e}")
                raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.load_model()

        try:
            embeddings_list = self.embeddings.embed_documents(texts)

            logger.info(f"Generated {len(embeddings_list)} OpenAI embeddings of dimension {len(embeddings_list[0])}")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise

    def encode_single(self, text: str) -> List[float]:
        """
        Generates embedding for a single text (query).

        Args:
            text: Text to generate embedding

        Returns:
            Embedding vector
        """
        self.load_model()

        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Query embedding generated: {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimates the number of tokens in a text.
        Uses conservative heuristic: ~3 characters per token (safer for technical text).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 3

    async def encode_async(self, texts: List[str], max_tokens_per_batch: int = 10000) -> List[List[float]]:
        """
        Asynchronous version of encode() with smart batching to avoid rate limits.

        Creates batches based on TOKEN COUNT (not just number of texts) to respect
        OpenAI's 40k tokens/min limit. Uses 10k tokens per batch for maximum safety.

        Args:
            texts: List of texts to generate embeddings
            max_tokens_per_batch: Maximum tokens per batch (default: 10000)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.load_model()

        # Estimate tokens for each text
        token_counts = [self._estimate_tokens(text) for text in texts]
        total_tokens = sum(token_counts)

        logger.info(f"Processing {len(texts)} texts (~{total_tokens:,} tokens total)")

        # Create batches based on token limits
        batches = []
        current_batch = []
        current_batch_tokens = 0

        for text, token_count in zip(texts, token_counts):
            # If adding this text would exceed limit, start new batch
            if current_batch and (current_batch_tokens + token_count > max_tokens_per_batch):
                batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = token_count
            else:
                current_batch.append(text)
                current_batch_tokens += token_count

        # Add last batch if not empty
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Split into {len(batches)} batches (max ~{max_tokens_per_batch:,} estimated tokens each)")

        # Process batches
        all_embeddings = []
        for batch_num, batch in enumerate(batches, 1):
            batch_tokens = sum(self._estimate_tokens(text) for text in batch)
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} texts, ~{batch_tokens:,} tokens)")

            try:
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Batch {batch_num}/{len(batches)} completed: {len(batch_embeddings)} embeddings")

                # Delay between batches to avoid rate limit (only if not the last batch)
                if batch_num < len(batches):
                    await asyncio.sleep(3.0)  # 3 seconds between batches

            except Exception as e:
                logger.error(f"Error in batch {batch_num}/{len(batches)}: {e}")
                raise

        logger.info(f"Total embeddings generated: {len(all_embeddings)} of dimension {len(all_embeddings[0])}")
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """
        Returns the embedding dimension of the model.

        Returns:
            Dimension of embedding vectors
        """
        # text-embedding-3-large tem dimensão 3072
        # text-embedding-3-small tem dimensão 1536
        if "large" in self.model_name:
            return 3072
        elif "small" in self.model_name:
            return 1536
        else:
            # Fallback: generate a test embedding to discover
            self.load_model()
            test_embedding = self.encode_single("test")
            return len(test_embedding)


# Global instance (singleton) of the embeddings service
_embeddings_service = None


def get_embeddings_service() -> EmbeddingsService:
    """
    Returns the singleton instance of the embeddings service.

    Returns:
        EmbeddingsService instance
    """
    global _embeddings_service

    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
        # Load model on initialization
        _embeddings_service.load_model()

    return _embeddings_service

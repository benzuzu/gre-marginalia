#!/usr/bin/env python3
"""
GRE Vocabulary Study Tool
Takes a book text file and intelligently replaces words with GRE vocabulary
based on semantic similarity.
"""

import os
import re
import csv
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import openai
from google import genai
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

class GREVocabTool:
    def __init__(self):
        # OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Gemini client for LLM calls
        self.genai_client = genai.Client()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.max_replacements = int(os.getenv("MAX_REPLACEMENTS_PER_PARAGRAPH"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
        
        # Store word definitions for quick lookup
        self.word_definitions = {}
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant='687e1305-5f97-43c3-8b0f-09adac8d4077',
            database='gre-study-dev'
        )
        
        # Get or create collection for GRE words
        self.collection = self.chroma_client.get_or_create_collection(
            name="gre_words_and_definitions",
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def load_gre_words(self, words_file: str):
        """Load GRE words from file and add to vector DB. Supports CSV (semicolon-delimited) and plain text."""
        print(f"Loading GRE words from {words_file}...")
        
        words = []
        word_definitions = {}  # Store word -> definition mapping
        
        # Check if it's a CSV file
        if words_file.lower().endswith('.csv'):
            with open(words_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # Skip header row
                for row in reader:
                    if row and row[0].strip():  # Get first column (word)
                        word = row[0].strip()
                        definition = row[1].strip() if len(row) > 1 and row[1].strip() else ""
                        words.append(word)
                        if definition:
                            word_definitions[word] = definition
                            self.word_definitions[word] = definition
        else:
            # Plain text file (one word per line)
            with open(words_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        
        # Check if collection already has words
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Found {existing_count} existing words in DB. Skipping reload.")
            return
        
        print(f"Adding {len(words)} words to vector DB...")
        
        # Process in batches to avoid exceeding ChromaDB's batch size limit (1000)
        batch_size = 200
        total_batches = (len(words) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(words))
            batch_words = words[start_idx:end_idx]
            
            print(f"Processing batch {batch_num + 1}/{total_batches} (words {start_idx + 1}-{end_idx})...")
            
            # Get embeddings for batch - use "word: definition" format for better semantic understanding
            batch_embeddings = []
            batch_metadatas = []
            for i, word in enumerate(batch_words):
                # Create embedding from word + definition for better semantic matching
                if word in word_definitions:
                    embedding_text = f"{word}: {word_definitions[word]}"
                else:
                    embedding_text = word
                
                embedding = self.get_embedding(embedding_text)
                batch_embeddings.append(embedding)
                
                # Store definition in metadata for future reference
                metadata = {"definition": word_definitions.get(word, "")} if word_definitions else {}
                batch_metadatas.append(metadata)
                
                if i % 50 == 0:
                    print(f"  Embedded word {start_idx + i + 1}/{len(words)}: {word}, {embedding[:3]}...")
            
            # Add batch to ChromaDB
            batch_ids = [f"word_{start_idx + i}" for i in range(len(batch_words))]
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_words,
                ids=batch_ids,
                metadatas=batch_metadatas if batch_metadatas[0] else None
            )
            
            print(f"  Added batch {batch_num + 1}/{total_batches} to vector DB.")
        
        print(f"Successfully added {len(words)} words to vector DB!")
    
    def split_into_paragraphs(self, text: str, chunk_size: int = 4) -> List[str]:
        """
        Split text into sentence chunks (grouped by 4 sentences).
        Note: Despite the function name, this actually splits into sentence chunks,
        not traditional paragraphs, to handle poorly formatted text files.
        """
        # Split text into sentences using common sentence endings
        # Handle abbreviations and decimal numbers
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences (remove extra whitespace)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def find_similar_words(self, paragraph: str, paragraph_index) -> List[Tuple[str, float]]:
        """Find GRE words similar to the paragraph context."""
        paragraph_embedding = self.get_embedding(paragraph)
        
        results = self.collection.query(
            query_embeddings=[paragraph_embedding],
            n_results=self.max_replacements # only get the necessary results
        )
        words_with_scores = []

        if paragraph_index == 0:
            print(results)
        
        # ChromaDB returns nested lists: results['documents'] = [['word1', 'word2', ...]]
        if results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]  # Get the inner list
            distances = results['distances'][0] if results.get('distances') else []
            metadatas = results['metadatas'][0] if results.get('metadatas') else []
            
            for idx, word in enumerate(documents):
                # Get distance (if available)
                distance = distances[idx] if idx < len(distances) else float('inf')
                
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1 - distance
                
                # Extract definition from metadata if available
                if idx < len(metadatas) and metadatas[idx]:
                    definition = metadatas[idx].get('definition', '')
                    if definition:
                        # Store definition in our dictionary for later use
                        self.word_definitions[word] = definition
                
                if similarity >= self.similarity_threshold:
                    print(f"Paragraph {paragraph_index} has similar word {word} (similarity: {similarity:.3f})!")
                    words_with_scores.append((word, similarity))
        
        return words_with_scores
    
    def add_gre_word_to_paragraph(self, paragraph: str, gre_word: str, definition: str = "") -> str:
        """Use Gemini to intelligently add a phrase or sentence using the GRE word."""
        # Get definition from metadata if available
        definition_text = f" (Definition: {definition})" if definition else ""
        
        prompt = f"""You are a writing assistant helping to enhance a paragraph with GRE vocabulary words.

Original paragraph:
{paragraph}

GRE word to incorporate: {gre_word}{definition_text}

Task: Add a phrase or sentence into this paragraph that naturally uses the word "{gre_word}". The addition should:
1. Fit seamlessly into the paragraph's style and tone
2. Make sense contextually
3. Use the word "{gre_word}" correctly
4. Enhance the paragraph without disrupting its flow

Return ONLY the enhanced paragraph with your addition. Do not include any explanations or notes."""

        try:
            response = self.genai_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            
            enhanced = response.text.strip()
            
            # Find and wrap the GRE word with asterisks (case-insensitive, whole word match)
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(gre_word) + r'\b'
            enhanced = re.sub(pattern, f'*{gre_word}*', enhanced, flags=re.IGNORECASE)
            
            print("MODIFIED PARAGRAPH:", enhanced, "\n-----------------------------------")
            return enhanced
        except Exception as e:
            print(f"Error calling Gemini for word '{gre_word}': {e}")
            return paragraph
    
    def replace_words_in_paragraph(self, paragraph: str, gre_words: List[Tuple[str, float]]) -> str:
        """Find the best matching GRE word and use GPT to add it naturally to the paragraph."""
        if not gre_words:
            return paragraph
        
        # Get the best matching word (highest similarity)
        best_word, best_similarity = gre_words[0]
        
        # Get definition from stored dictionary
        definition = self.word_definitions.get(best_word, "")
        
        # Use GPT to add the word naturally
        enhanced_paragraph = self.add_gre_word_to_paragraph(paragraph, best_word, definition)

        return enhanced_paragraph
    
    def process_book(self, input_file: str, output_file: str, gre_words_file: str):
        """Process a book file and create GRE-enhanced version."""
        # Load GRE words into vector DB
        self.load_gre_words(gre_words_file)
        
        # Read input book
        print(f"\nReading book from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            book_text = f.read()
        
        # Split into sentence chunks (4 sentences per chunk)
        # Note: split_into_paragraphs actually splits into 4-sentence chunks
        sentence_chunks = self.split_into_paragraphs(book_text, chunk_size=4)
        print(f"Found {len(sentence_chunks)} chunks (4 sentences each).")
        
        # Process each chunk
        enhanced_chunks = []
        for i, chunk in enumerate(sentence_chunks):
            if (i + 1) % 10 == 0:
                print(f"Processing chunk {i+1}/{len(sentence_chunks)}...")
            
            # Find similar GRE words
            similar_words = self.find_similar_words(chunk, i)
            
            # Enhance chunk if we found good matches
            if similar_words:
                enhanced_chunk = self.replace_words_in_paragraph(chunk, similar_words)
                enhanced_chunks.append(enhanced_chunk)
            else:
                enhanced_chunks.append(chunk)
        
        # Write output
        print(f"\nWriting enhanced book to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(enhanced_chunks))
        
        print("Done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRE Vocabulary Study Tool')
    parser.add_argument('input_file', help='Input book text file')
    parser.add_argument('--gre-words', default='vocab_mixed.csv', help='File containing GRE words (CSV with semicolon delimiter or plain text, one per line)')
    parser.add_argument('--output', default='enhanced_book.txt', help='Output file for enhanced book')
    
    args = parser.parse_args()
    
    tool = GREVocabTool()
    tool.process_book(args.input_file, args.output, args.gre_words)


if __name__ == "__main__":
    main()


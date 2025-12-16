# GRE Vocabulary Study Tool

A fun tool that intelligently inserts GRE vocabulary words into your favorite books based on semantic similarity. Read The Great Gatsby (or any book) while learning GRE words!

## How It Works

1. **Text Processing**: Splits your book into paragraphs
2. **Embeddings**: Uses OpenAI's embedding API to create vector representations of each paragraph
3. **Semantic Matching**: Cross-references paragraph embeddings with a vector database of GRE words
4. **Smart Insertion**: Inserts appropriate words in paragraphs with semantically similar GRE vocabulary, via an in-context sentence or phrase.

### Example (Great Gatsby)

## Setup

1. **Install dependencies**:

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
   Create a `.env` file in the project root. `.env.example` has all the required variables. You will need to setup openai, gemini, and chromadb accounts.

3. **Prepare your files**:

- `vocab_mixed.csv`: Contains GRE words in CSV format (semicolon-delimited, word;definition) - already included
- Your book text file: A `.txt` file of the book you want to enhance. Examples are provided in `books/`

## Usage

```bash
python main.py books/your_book.txt
```

Options:

- `--gre-words`: Path to GRE words file (default: `vocab_mixed.csv`). Supports CSV (semicolon-delimited) or plain text (one word per line)
- `--output`: Output file path (default: `modifie_books/<input_file>`)

## Example

## How It Works Internally

1. **Vector Database**: Uses ChromaDB to store GRE word embeddings locally
2. **Embedding Model**: Uses OpenAI's `text-embedding-3-small` (configurable)
3. **Similarity Search**: Finds GRE words semantically similar to each paragraph
4. **Replacement Logic**: Replaces words that are 4+ characters and not common stop words

## Configuration

Edit `.env` to adjust:

- `MAX_REPLACEMENTS_PER_PARAGRAPH`: How many words to replace per paragraph (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for replacement (-1 -> 1, default -.3)

## Notes

- The first run will take longer as it builds the vector database
- Subsequent runs reuse the existing database
- Processing time depends on book length and API rate limits
- Make sure you have sufficient OpenAI, Gemini, ChromaDB API credits

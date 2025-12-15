# GRE Vocabulary Study Tool

A fun tool that intelligently inserts GRE vocabulary words into your favorite books based on semantic similarity. Read Harry Potter (or any book) while learning GRE words!

## How It Works

1. **Text Processing**: Splits your book into paragraphs
2. **Embeddings**: Uses OpenAI's embedding API to create vector representations of each paragraph
3. **Semantic Matching**: Cross-references paragraph embeddings with a vector database of GRE words
4. **Smart Replacement**: Replaces appropriate words in paragraphs with semantically similar GRE vocabulary

## Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
   Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
MAX_REPLACEMENTS_PER_PARAGRAPH=3
SIMILARITY_THRESHOLD=0.7
```

3. **Prepare your files**:

- `vocab_mixed.csv`: Contains GRE words in CSV format (semicolon-delimited, word;definition) - already included
- Your book text file: A `.txt` file of the book you want to enhance

## Usage

```bash
python gre_vocab_tool.py your_book.txt --output enhanced_book.txt
```

Options:

- `--gre-words`: Path to GRE words file (default: `vocab_mixed.csv`). Supports CSV (semicolon-delimited) or plain text (one word per line)
- `--output`: Output file name (default: `enhanced_book.txt`)

## Example

```bash
python gre_vocab_tool.py harry_potter.txt --output harry_potter_gre.txt
```

## How It Works Internally

1. **Vector Database**: Uses ChromaDB to store GRE word embeddings locally
2. **Embedding Model**: Uses OpenAI's `text-embedding-3-small` (configurable)
3. **Similarity Search**: Finds GRE words semantically similar to each paragraph
4. **Replacement Logic**: Replaces words that are 4+ characters and not common stop words

## Configuration

Edit `.env` to adjust:

- `MAX_REPLACEMENTS_PER_PARAGRAPH`: How many words to replace per paragraph (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for replacement (0.0-1.0, default: 0.7)
- `EMBEDDING_MODEL`: OpenAI embedding model to use

## Alternative Study Ideas

Here are some other fun ways to study GRE vocabulary:

1. **Flashcard Generator**: Create Anki decks from your word list
2. **Context Clue Trainer**: Generate sentences with GRE words and practice inferring meanings
3. **Word Association Game**: Build a graph of semantically related words
4. **Daily Reading Challenge**: Replace words in news articles or blog posts
5. **Story Generator**: Use GPT to create short stories using specific GRE words
6. **Spaced Repetition Bot**: A chatbot that quizzes you on words at optimal intervals
7. **Pronunciation Practice**: Audio flashcards with word definitions
8. **Etymology Explorer**: Visualize word roots and origins
9. **Synonym/Antonym Trainer**: Practice finding relationships between words
10. **Writing Prompts**: Generate prompts that require using specific GRE words

## Notes

- The first run will take longer as it builds the vector database
- Subsequent runs reuse the existing database
- Processing time depends on book length and API rate limits
- Make sure you have sufficient OpenAI API credits

"""Small knowledge base used by the semantic-search demonstration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One searchable document stored in the in-memory vector store."""

    document_id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()


SAMPLE_DOCUMENTS = (
    Document(
        document_id="attention",
        title="Self-Attention in Transformers",
        text=(
            "Self-attention lets every token examine other tokens in the same "
            "sequence. Query, key, and value projections produce attention "
            "weights that decide which token relationships matter most."
        ),
        tags=("transformers", "attention", "nlp"),
    ),
    Document(
        document_id="embeddings",
        title="Text Embeddings",
        text=(
            "An embedding converts text into a numeric vector that represents "
            "meaning. Similar sentences receive vectors that point in similar "
            "directions, enabling semantic similarity and retrieval."
        ),
        tags=("embeddings", "vectors", "semantic-search"),
    ),
    Document(
        document_id="rag",
        title="Retrieval-Augmented Generation",
        text=(
            "RAG retrieves relevant knowledge before asking a language model to "
            "generate an answer. Retrieved context helps ground the response in "
            "documents and can reduce unsupported claims."
        ),
        tags=("rag", "retrieval", "llm"),
    ),
    Document(
        document_id="tokenization",
        title="Tokenization for Language Models",
        text=(
            "Tokenization divides text into reusable units such as words, "
            "subwords, punctuation, and numbers. BPE, WordPiece, and "
            "SentencePiece use different strategies to create tokens."
        ),
        tags=("tokenization", "bpe", "nlp"),
    ),
    Document(
        document_id="cosine",
        title="Cosine Similarity",
        text=(
            "Cosine similarity compares the direction of two vectors. A higher "
            "score means the vectors point in more similar directions and the "
            "represented texts are likely more closely related."
        ),
        tags=("similarity", "vectors", "math"),
    ),
    Document(
        document_id="vector-store",
        title="Vector Stores",
        text=(
            "A vector store keeps document embeddings and searches them by "
            "similarity. Production systems may use approximate nearest-neighbor "
            "indexes, metadata filters, persistence, and distributed storage."
        ),
        tags=("database", "vector-store", "retrieval"),
    ),
    Document(
        document_id="training",
        title="Neural Network Training",
        text=(
            "Training adjusts model parameters by measuring loss, computing "
            "gradients with backpropagation, and applying an optimizer over many "
            "batches and epochs."
        ),
        tags=("training", "backpropagation", "optimization"),
    ),
    Document(
        document_id="docker",
        title="Containerized ML Deployment",
        text=(
            "Docker packages an application and its dependencies into a "
            "container image. Containers make machine-learning services easier "
            "to reproduce and deploy to cloud infrastructure."
        ),
        tags=("docker", "containers", "deployment"),
    ),
    Document(
        document_id="testing",
        title="Testing Machine-Learning Software",
        text=(
            "Unit tests validate small functions such as similarity calculations. "
            "Integration tests verify that embedding, storage, retrieval, and "
            "ranking work together correctly."
        ),
        tags=("testing", "pytest", "quality"),
    ),
    Document(
        document_id="encoder-decoder",
        title="Encoder-Decoder Transformers",
        text=(
            "An encoder reads an input sequence and builds contextual "
            "representations. A decoder generates output tokens while attending "
            "to both previous output tokens and the encoder representations."
        ),
        tags=("transformers", "encoder", "decoder"),
    ),
)

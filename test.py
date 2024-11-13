# FinDER for example.
# You can use other tasks such as `FinQA`, `TATQA`, etc.
from financerag.tasks import FinDER
finder_task = FinDER()

from sentence_transformers import SentenceTransformer, CrossEncoder
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval

# model = SentenceTransformer('intfloat/e5-large-v2')
# We need to put prefix for e5 models.
# For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
encoder_model = SentenceTransformerEncoder(
    model_name_or_path='intfloat/e5-large-v2',
    # q_model=model,
    # doc_model=model,
    query_prompt='query: ',
    doc_prompt='passage: '
)
retriever = DenseRetrieval(model=encoder_model)

# Retrieve relevant documents
results = finder_task.retrieve(retriever=retriever)

# Rerank the results
from financerag.rerank import CrossEncoderReranker
reranker = CrossEncoderReranker(CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'))
reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)

finder_task.save_results(output_dir='output/')
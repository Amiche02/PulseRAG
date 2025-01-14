import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from workflow import ExtractionIndexingWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_indexed_data_as_markdown(indexed_data, output_dir):
    """
    Saves the indexed data into two separate Markdown (.md) files:
      1. search_results.md  --> The chunk text and metadata
      2. embeddings.md      --> The chunk embeddings and metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    search_results_path = os.path.join(output_dir, "search_results.md")
    embeddings_path = os.path.join(output_dir, "embeddings.md")

    with open(search_results_path, "w", encoding="utf-8") as sr_file:
        sr_file.write("# Search Results\n\n")
        for doc in indexed_data:
            sr_file.write(f"## Document: {doc['document_id']}\n\n")
            for chunk in doc.get("chunks", []):
                sr_file.write(f"### Chunk #{chunk['chunk_index']}\n")
                sr_file.write(f"{chunk['content']}\n")
                sr_file.write(f"**Metadata:** {json.dumps(chunk['metadata'], ensure_ascii=False)}\n\n")

    with open(embeddings_path, "w", encoding="utf-8") as emb_file:
        emb_file.write("# Embeddings\n\n")
        for doc in indexed_data:
            emb_file.write(f"## Document: {doc['document_id']}\n\n")
            for chunk in doc.get("chunks", []):
                emb_file.write(f"### Chunk #{chunk['chunk_index']} Embedding\n")
                emb_file.write(f"{chunk['embedding']}\n")
                emb_file.write(f"**Metadata:** {json.dumps(chunk['metadata'], ensure_ascii=False)}\n\n")

    logger.info(f"Markdown files saved to '{output_dir}'.")

def save_indexed_data_as_json(indexed_data, output_dir):
    """
    Saves indexed data into two separate JSON files:
      1. search_results.json --> chunk text & metadata (no embeddings)
      2. embeddings.json     --> embeddings (without full text)
    """
    os.makedirs(output_dir, exist_ok=True)
    results_data = []
    embeddings_data = []

    for doc in indexed_data:
        doc_id = doc["document_id"]
        results_chunks = []
        embeddings_chunks = []

        for chunk in doc.get("chunks", []):
            results_chunks.append({
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            })
            embedding = chunk["embedding"]
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            embeddings_chunks.append({
                "chunk_index": chunk["chunk_index"],
                "embedding": embedding
            })

        results_data.append({"document_id": doc_id, "chunks": results_chunks})
        embeddings_data.append({"document_id": doc_id, "chunks": embeddings_chunks})

    results_path = os.path.join(output_dir, "search_results.json")
    embeddings_path = os.path.join(output_dir, "embeddings.json")

    with open(results_path, "w", encoding="utf-8") as f_res:
        json.dump(results_data, f_res, indent=2, ensure_ascii=False)

    with open(embeddings_path, "w", encoding="utf-8") as f_emb:
        json.dump(embeddings_data, f_emb, indent=2, ensure_ascii=False)

    logger.info(f"Saved search results JSON to: {results_path}")
    logger.info(f"Saved embeddings JSON to: {embeddings_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process and index documents.")
    parser.add_argument(
        "--docs", nargs="+", required=True,
        help="List of document file paths to process."
    )
    parser.add_argument(
        "--save-format", choices=["json", "md"], default="json",
        help="Format to save extraction results."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/docs",
        help="Directory to save the output files."
    )
    return parser.parse_args()

async def main():
    args = parse_args()

    documents = []
    for idx, doc_path in enumerate(args.docs):
        documents.append({
            "document_id": f"doc{idx+1}",
            "file_path": doc_path,
            "metadata": {"title": os.path.basename(doc_path)}
        })

    extraction_workflow = ExtractionIndexingWorkflow()
    indexed_docs = await extraction_workflow.process_documents(documents)

    if args.save_format == "json":
        save_indexed_data_as_json(indexed_docs, output_dir=args.output_dir)
    else:
        save_indexed_data_as_markdown(indexed_docs, output_dir=args.output_dir)

if __name__ == "__main__":
    asyncio.run(main())

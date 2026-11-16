import argparse
from pathlib import Path

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.config import Config
from flipkart.data_convertor import DataConvertor

# Stores the index of the next document to upload, so a re-run resumes automatically.
PROGRESS_FILE = Path(__file__).resolve().parent / "ingest_progress.txt"
BATCH_SIZE = 500


def read_progress() -> int:
    try:
        return int(PROGRESS_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def write_progress(next_index: int) -> None:
    PROGRESS_FILE.write_text(str(next_index))


class DataIngestor:

    def __init__(self):

        # 1. Load embedding model
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

        # 2. Connect to Astra DB
        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, load_existing=True, start_from=0):

        if load_existing:
            return self.vstore

        # 3. Locate both CSV files
        data_path = Path(__file__).resolve().parent.parent / "data"

        review_csv = data_path / "flipkart_product_review_cleaned.csv"
        catalog_csv = data_path / "flipkard.csv"

        # 4. Convert both datasets into Documents
        converter = DataConvertor(
            str(review_csv),
            str(catalog_csv)
        )

        docs = converter.convert()
        total_docs = len(docs)

        print("Total documents created:", total_docs)

        if start_from >= total_docs:
            print("Nothing left to upload (start_from >= total documents).")
            return self.vstore

        if start_from > 0:
            print(f"Resuming from document index {start_from}")

        # 5. Upload documents in batches, saving progress after each one
        for start in range(start_from, total_docs, BATCH_SIZE):

            batch = docs[start:start + BATCH_SIZE]
            ids = [doc.id for doc in batch]

            self.vstore.add_documents(
                documents=batch,
                ids=ids
            )

            next_index = start + len(batch)
            write_progress(next_index)

            print(f"Uploaded {next_index}/{total_docs} documents")

        print("Total documents uploaded:", total_docs)

        return self.vstore


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Document index to resume from. Overrides the progress file.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the Astra DB collection and restart from 0.",
    )
    args = parser.parse_args()

    print("Starting data ingestion...")

    ingestor = DataIngestor()

    if args.clear:
        print("Clearing existing Astra DB collection...")
        ingestor.vstore.clear()
        PROGRESS_FILE.unlink(missing_ok=True)
        print("Collection cleared!")

    # Priority: --start-from, then the progress file, then 0
    if args.start_from is not None:
        start_from = args.start_from
    else:
        start_from = read_progress()

    ingestor.ingest(load_existing=False, start_from=start_from)

    print("Data ingestion completed successfully!")

    
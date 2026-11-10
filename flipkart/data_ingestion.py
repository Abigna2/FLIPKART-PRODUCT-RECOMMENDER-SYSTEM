
from pathlib import Path

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.config import Config
from flipkart.data_convertor import DataConvertor


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

    def ingest(self, load_existing=True):

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

        print("Total documents created:", len(docs))

        # 5. Upload documents in batches
        batch_size = 500

        total_uploaded = 0

        for start in range(0, len(docs), batch_size):

            batch = docs[start:start + batch_size]

            ids = [doc.id for doc in batch]

            self.vstore.add_documents(
                documents=batch,
                ids=ids
            )

            total_uploaded += len(batch)

            print(
                f"Uploaded {total_uploaded}/{len(docs)} documents"
            )

        print("Total documents uploaded:", total_uploaded)

        return self.vstore


if __name__ == "__main__":

    print("Starting data ingestion...")

    ingestor = DataIngestor()

    # WARNING: Clears the entire collection
    print("Clearing existing Astra DB collection...")
    ingestor.vstore.clear()

    print("Collection cleared!")

    # Ingest both datasets
    vstore = ingestor.ingest(load_existing=False)

    print("Data ingestion completed successfully!")

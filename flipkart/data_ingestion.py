from pathlib import Path

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.config import Config
from flipkart.data_convertor import DataConvertor


class DataIngestor:

    def __init__(self):

        # 1. Load the Hugging Face embedding model locally
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

        # 3. If an existing vector store is requested, return it
        if load_existing:
            return self.vstore

        # 4. Locate the CSV file
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "flipkart_product_review.csv"
        )

        # 5. Convert CSV data into LangChain Documents
        docs = DataConvertor(str(csv_path)).convert()

        print("Total documents created:", len(docs))

        # 6. Upload documents and embeddings to Astra DB
        ids = self.vstore.add_documents(docs)

        print("Total documents uploaded:", len(ids))

        return self.vstore


# 7. Execute ingestion only when this file is run directly

if __name__ == "__main__":
    print("Starting data cleanup...")

    ingestor = DataIngestor()

    # STEP 1: Remove all existing records (1350 duplicates included)
    print("Clearing existing Astra DB collection...")
    ingestor.vstore.clear()
    print("Collection cleared!")

    # STEP 2: Load the original CSV and ingest it once
    print("Uploading original 450 reviews...")
    vstore = ingestor.ingest(load_existing=False)

    print("Data cleanup completed!")

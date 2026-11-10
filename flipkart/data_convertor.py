import pandas as pd
from langchain_core.documents import Document
from hashlib import sha256

class DataConvertor:

    def __init__(self, review_file_path: str, catalog_file_path: str):
        self.review_file_path = review_file_path
        self.catalog_file_path = catalog_file_path

    def convert(self):
        # DATASET 1: CUSTOMER REVIEWS
        review_df = pd.read_csv(self.review_file_path)

        review_df = review_df[["product_title", "rating", "summary", "review"]].copy()

        review_df = review_df.dropna(subset=["product_title", "review"])

        review_df["product_title"] = review_df["product_title"].astype(str).str.strip()
        review_df["review"] = review_df["review"].astype(str).str.strip()

        review_df = review_df[
            (review_df["product_title"] != "") &
            (review_df["review"] != "")
        ]

        review_df = review_df.drop_duplicates(
            subset=["product_title", "review"]
        )

        review_docs = []

        for _, row in review_df.iterrows():

            product_name = row["product_title"]
            review = row["review"]

            unique_text = "review|" + product_name + "|" + review
            doc_id = sha256(unique_text.encode("utf-8")).hexdigest()

            doc = Document(
                id=doc_id,
                page_content=f"""
Product Name: {product_name}
Customer Rating: {row["rating"]}
Review Summary: {row["summary"]}
Customer Review: {review}
""",
                metadata={
                    "source": "customer_review",
                    "product_name": product_name,
                    "rating": int(row["rating"]) if pd.notna(row["rating"]) else 0,
                    "summary": str(row["summary"]) if pd.notna(row["summary"]) else ""
                }
            )

            review_docs.append(doc)
        # DATASET 2: PRODUCT CATALOG
        catalog_df = pd.read_csv(self.catalog_file_path)

        catalog_docs = []

        for _, row in catalog_df.iterrows():

            product_name = str(row["product_name"])
            product_id = str(row["product_id"])

            unique_text = "catalog|" + product_id
            doc_id = sha256(unique_text.encode("utf-8")).hexdigest()

            doc = Document(
                id=doc_id,
                page_content=f"""
Product Name: {product_name}
Category: {row["category"]}
Brand: {row["brand"]}
Price: {row["price"]}
Discount Percentage: {row["discount_percent"]}
Final Price: {row["final_price"]}
Rating: {row["rating"]}
Review Count: {row["review_count"]}
Stock Available: {row["stock_available"]}
Seller: {row["seller"]}
Seller City: {row["seller_city"]}
Delivery Days: {row["delivery_days"]}
Warranty Months: {row["warranty_months"]}
Color: {row["color"]}
Size: {row["size"]}
Return Policy Days: {row["return_policy_days"]}
""",
                metadata={
                    "source": "product_catalog",
                    "product_id": product_id,
                    "product_name": product_name,
                    "category": str(row["category"]),
                    "brand": str(row["brand"]),
                    "price": float(row["price"]),
                    "final_price": float(row["final_price"]),
                    "rating": float(row["rating"]),
                    "stock_available": int(row["stock_available"])
                }
            )

            catalog_docs.append(doc)
        # COMBINE BOTH DATASETS
        docs = review_docs + catalog_docs

        print("Unique customer review documents:", len(review_docs))
        print("Product catalog documents:", len(catalog_docs))
        print("Total documents created:", len(docs))

        return docs

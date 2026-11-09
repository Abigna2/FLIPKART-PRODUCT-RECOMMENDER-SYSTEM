
from langchain_groq import ChatGroq
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config


class RAGChainBuilder:

    def __init__(self, vector_store):
        self.vector_store = vector_store

        self.model = ChatGroq(
            model=Config.RAG_MODEL,
            temperature=0.2
        )

        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()

        return self.history_store[session_id]

    def build_chain(self):

        # Retrieve relevant product reviews
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        # DEBUG: Print retrieved documents in terminal
        print("\n========== RETRIEVED PRODUCT REVIEWS ==========")
        print("DEBUG: Starting document retrieval...", flush=True)
        
        docs = retriever.invoke(
            "Suggest me some great headphones"
        )

        print(f"DEBUG: Retrieved {len(docs)} documents", flush=True)

        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}")
            print("Metadata:", doc.metadata)
            print("Content:", doc.page_content)

        print("\n========== END OF RETRIEVED REVIEWS ==========\n")

        # Rewrite follow-up questions into standalone questions
        context_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Given the chat history and the latest user question,
                rewrite the question into a standalone question.

                Do not answer the question.
                Preserve the user's product preferences and requirements."""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Strict review-grounded QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are Flipkart Chatbot, an e-commerce product
                recommendation assistant.

                Your job is to answer product-related questions using
                ONLY the retrieved product titles and customer reviews.

                STRICT RULES:

                1. Recommend only products whose names are explicitly
                   present in the retrieved context.

                2. Never invent product names, specifications,
                   features, prices, ratings, or review details.

                3. Every recommendation must be supported by information
                   from the retrieved reviews or product titles.

                4. Explain briefly why each recommended product is
                   suitable based on the available review evidence.

                5. If the user asks for gaming headsets but the context
                   contains only ordinary headphones or unrelated
                   products, clearly state that relevant gaming headset
                   review data was not found.

                6. Do not use your general knowledge to fill gaps
                   in the retrieved context.

                7. If the retrieved context is insufficient, say:
                   "I couldn't find enough relevant product review
                   data to make a reliable recommendation."

                8. Be concise, friendly, and helpful.

                9. If the user specifies a budget or preference,
                   recommend only products supported by the context
                   that match those requirements.

                10. Do not claim a product is the best, highly rated,
                    or well-reviewed unless the context supports it.

                CONTEXT:
                {context}
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.model,
            retriever,
            context_prompt
        )

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            self.model,
            qa_prompt
        )

        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Add conversation memory
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

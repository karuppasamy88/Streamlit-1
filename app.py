import os
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Intent(str, Enum):
    TECHNICAL_ISSUE = "TECHNICAL_ISSUE"
    REGISTER_COMPLAINT = "REGISTER_COMPLAINT"
    CHECK_STATUS = "CHECK_STATUS"
    GENERAL_QUERY = "GENERAL_QUERY"


class SupportRAGSystem:
    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base",
        chroma_dir: str = "chroma_db",
        sqlite_db_path: str = "support_tickets.db",
        model_name: str = "gpt-4o-mini",
    ):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chroma_dir = Path(chroma_dir)
        self.sqlite_db_path = sqlite_db_path
        self.model_name = model_name

        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()

        self._init_sqlite()
        self.vectorstore = self._init_chroma()

    # ------------------------------
    # Database methods
    # ------------------------------
    def _init_sqlite(self) -> None:
        """Initialize SQLite database and complaints table."""
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def register_complaint(self, user: str, issue: str) -> int:
        """Insert a new complaint and return generated ticket id."""
        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO complaints (user, issue, status, timestamp) VALUES (?, ?, ?, ?)",
                (user, issue, "Pending", timestamp),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def check_ticket_status(self, ticket_id: int) -> Optional[Tuple[int, str, str, str, str]]:
        """Fetch complaint by id. Returns None if not found."""
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user, issue, status, timestamp FROM complaints WHERE id = ?",
                (ticket_id,),
            )
            row = cursor.fetchone()
            return row

    # ------------------------------
    # Knowledge base + retrieval
    # ------------------------------
    def _load_docs(self) -> List[Document]:
        """Load documents from the knowledge_base folder."""
        docs: List[Document] = []
        if not self.knowledge_base_dir.exists():
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            return docs

        for file_path in self.knowledge_base_dir.iterdir():
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs.extend(loader.load())
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs.extend(loader.load())
        return docs

    def _init_chroma(self) -> Chroma:
        """Initialize a persistent Chroma vector DB and ingest if empty."""
        vectorstore = Chroma(
            collection_name="support_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_dir),
        )

        existing_count = vectorstore._collection.count()  # pylint: disable=protected-access
        if existing_count == 0:
            docs = self._load_docs()
            if docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                )
                chunks = splitter.split_documents(docs)
                vectorstore.add_documents(chunks)
        return vectorstore

    def retrieve_technical_context(self, query: str, k: int = 3) -> List[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)

    # ------------------------------
    # Router + handlers
    # ------------------------------
    def classify_intent(self, user_message: str) -> Intent:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an intent classifier for a customer support system.
                    Return only one label from this exact list:
                    TECHNICAL_ISSUE, REGISTER_COMPLAINT, CHECK_STATUS, GENERAL_QUERY.

                    Rules:
                    - TECHNICAL_ISSUE: troubleshooting/product/how-to problems that should search manuals/docs.
                    - REGISTER_COMPLAINT: user is reporting a complaint/problem and wants to raise a ticket.
                    - CHECK_STATUS: user asks for complaint/ticket status and may include ticket id.
                    - GENERAL_QUERY: greetings, chitchat, or general questions not needing docs/db.

                    Output exactly one label and nothing else.
                    """,
                ),
                ("human", "Message: {message}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        label = chain.run(message=user_message).strip().upper()
        try:
            return Intent(label)
        except ValueError:
            return Intent.GENERAL_QUERY

    def answer_technical_issue(self, user_message: str) -> str:
        docs = self.retrieve_technical_context(user_message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        if not context.strip():
            return "I cannot find this in the manual."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a technical support assistant.
                    Answer strictly from the provided context.
                    If the answer cannot be found in the context, reply exactly:
                    I cannot find this in the manual.
                    """,
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(question=user_message, context=context).strip()

    def extract_complaint_fields(self, user_message: str) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Extract the user's name and complaint issue description.
                    Return only JSON with keys: user, issue.
                    If name is missing, set user to "Unknown".
                    """,
                ),
                ("human", "Message: {message}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        raw = chain.run(message=user_message).strip()

        # Lightweight JSON parsing fallback
        user = "Unknown"
        issue = user_message
        try:
            import json

            data = json.loads(raw)
            user = str(data.get("user") or "Unknown").strip()
            issue = str(data.get("issue") or user_message).strip()
        except Exception:
            pass

        return {"user": user, "issue": issue}

    def extract_ticket_id(self, user_message: str) -> Optional[int]:
        import re

        match = re.search(r"\b(\d{1,10})\b", user_message)
        if not match:
            return None
        return int(match.group(1))

    def handle_message(self, user_message: str) -> str:
        intent = self.classify_intent(user_message)

        if intent == Intent.TECHNICAL_ISSUE:
            return self.answer_technical_issue(user_message)

        if intent == Intent.REGISTER_COMPLAINT:
            fields = self.extract_complaint_fields(user_message)
            ticket_id = self.register_complaint(fields["user"], fields["issue"])
            return (
                f"Complaint registered successfully. "
                f"Your Ticket ID is #{ticket_id}. Current status: Pending."
            )

        if intent == Intent.CHECK_STATUS:
            ticket_id = self.extract_ticket_id(user_message)
            if ticket_id is None:
                return "Please provide a valid Ticket ID so I can check the complaint status."
            row = self.check_ticket_status(ticket_id)
            if row is None:
                return f"Ticket ID #{ticket_id} was not found."
            _, user, issue, status, timestamp = row
            return (
                f"Ticket #{ticket_id} for {user}:\n"
                f"Issue: {issue}\n"
                f"Status: {status}\n"
                f"Created at: {timestamp}"
            )

        # GENERAL_QUERY
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a friendly customer support assistant."),
                ("human", "{message}"),
            ]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(message=user_message).strip()


def main() -> None:
    st.set_page_config(page_title="Customer Support RAG", page_icon="🛠️")
    st.title("🛠️ Customer Support RAG System")
    st.caption("Router-driven support: Technical Docs, Complaint Registration, Status Tracking, and General Chat")

    if "assistant" not in st.session_state:
        st.session_state.assistant = SupportRAGSystem()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I can help with technical issues, register complaints, check ticket status, or answer general queries.",
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            response = st.session_state.assistant.handle_message(user_input)
        except Exception as exc:  # Generic catch for a user-friendly UI fallback
            response = f"Sorry, I ran into an error while processing your request: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()

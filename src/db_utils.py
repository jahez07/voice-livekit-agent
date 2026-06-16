import psycopg2
import sys
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def open_connection():
    conn = psycopg2.connect(
        database = 'rag',
        user = 'postgres',
        host = 'localhost',
        password = 'postgres',
        port = 5435
    )
    return conn

def get_embeddings(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding

    return embedding

def insert_meeting_recording(meeting_title, context):
    """
    Insert a meeting transcript with its embedding into pgvector for RAG retrieval.
    Args: 
        meeting_title: The title or name of the meeting, e.g. 'Q3 Revenue Review',
        context: The transcript or text content from the meeting to store
    """
    conn = open_connection()
    cursor = conn.cursor()
    meeting_title_emb = get_embeddings(meeting_title)
    embeddings = get_embeddings(context)
    cursor.execute(
        """
        INSERT INTO meeting_recording(
            context, meeting_title, meeting_title_emb, embedding
        ) VALUES (%s, %s, %s, %s)
        """, (
            context, meeting_title, 
            meeting_title_emb, embeddings
        )
    )
    
    conn.commit()
    cursor.close()
    conn.close()

    return f"Stored meeting '{meeting_title}' in the database"
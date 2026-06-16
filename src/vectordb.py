import psycopg2
import sys
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

conn = psycopg2.connect(
    database = 'rag',
    user = 'postgres',
    host = 'localhost',
    password = 'postgres',
    port = 5435
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

cursor.execute(
    """CREATE TABLE IF NOT EXISTS meeting_recording(
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    context TEXT,
    meeting_id VARCHAR NOT NULL,
    embedding VECTOR(1536)
    );"""
)

# Evaluating embedding insertion
text = "The Q3 revenue was $4.2M, up 15% from last quarter."
meeting_id = "meeting_001"

response = client.embeddings.create(
    input=text,
    model='text-embedding-3-small'
)

embedding = response.data[0].embedding

cursor.execute(
    """
    INSERT INTO meeting_recording(
        context, meeting_id, embedding
    ) VALUES (%s, %s, %s)
    """, (
        text, meeting_id, embedding
    )
)

conn.commit()
cursor.close()
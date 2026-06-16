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

def delete_table(conn, table_name):
    cursor = conn.cursor()
    table_name = str(table_name)
    cursor.execute(
        f"""
        DROP TABLE {table_name}
        """
    )

    conn.commit()
    cursor.close()

cursor.execute(
    """CREATE TABLE IF NOT EXISTS meeting_recording(
    meeting_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    context TEXT,
    meeting_title VARCHAR(50),
    meeting_title_emb VECTOR(1536),
    embedding VECTOR(1536)
    );"""
)

# Evaluating embedding insertion
text = "The Q3 revenue was $4.2M, up 15% from last quarter."
meeting_title = "test meeting"

response = client.embeddings.create(
    input=text,
    model='text-embedding-3-small'
)

embedding = response.data[0].embedding

cursor.execute(
    """
    INSERT INTO meeting_recording(
        context, embedding
    ) VALUES (%s, %s)
    """, (
        text, embedding
    )
)

conn.commit()
cursor.close()
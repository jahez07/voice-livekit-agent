import logging
import datetime
import time
from psycopg2 import pool
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

DB_HOST = 'pgvector'
DB_PORT = 5432
DB_NAME = 'rag'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

# Global DB pool
DB_POOL = None

# Initialize Connection Pool

def init_db_pool(minconn=5, maxconn=20, retries=5, delay=3):
    global DB_POOL
    attempt = 0
    while attempt < retries:
        try:
            DB_POOL = pool.ThreadedConnectionPool(
                minconn, maxconn,
                host=DB_HOST, port=DB_PORT,
                dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
            )
            logger.info("Database connection pool created successfully")
            return
        except Exception as e:
            attempt += 1
            logger.warning(f"DB pool init failed (attempt {attempt}/{retries}): {e} ")
            time.sleep(delay)
    logger.error("Failed to initialize DB pool after retries")
    raise ConnectionError("Cannot connect to DB")


# Get connection from pool

def get_connection():
    global DB_POOL
    if DB_POOL is None:
        init_db_pool()
    try:
        return DB_POOL.getconn()
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        return None
    
# Return to connection pool

def release_connection(conn):
    global DB_POOL
    if conn and DB_POOL:
        DB_POOL.putconn(conn)
    
def close_db_pool():
    global DB_POOL
    if DB_POOL:
        DB_POOL.closeall()
        logger.info("Database connection pool closed")


# Table creation functions
def create_meeting_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_data(
                       meeting_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                       meeting_tag VARCHAR(20) NOT NULL,
                       meeting_name VARCHAR(100),
            );
        """)
        conn.commit()
        cursor.close()
        logger.info("Created meeting_data table")
    except Exception as e:
        logger.error(f"Error creating meeting_data table: {e}")
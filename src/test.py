from dotenv import load_dotenv
import os

load_dotenv(".env")
elevan_labs = os.getenv("ELEVANLABS_API_KEY")
print(elevan_labs)
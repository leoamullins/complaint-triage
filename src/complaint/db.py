import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():
    return create_engine(os.environ["DATABASE_URL"])

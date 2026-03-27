import datetime
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy import inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_PATH = Path(__file__).with_name("shield_saas.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    subscription_end = Column(DateTime, nullable=True)
    current_otp = Column(String, nullable=True)
    current_otp_hash = Column(String, nullable=True)
    otp_expiry = Column(DateTime, nullable=True)
    otp_attempts = Column(Integer, default=0, nullable=False)
    otp_last_sent_at = Column(DateTime, nullable=True)
    last_payment_ref = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


def ensure_user_table_columns() -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("users")}
    statements = []

    if "current_otp_hash" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN current_otp_hash VARCHAR")
    if "otp_attempts" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN otp_attempts INTEGER NOT NULL DEFAULT 0")
    if "otp_last_sent_at" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN otp_last_sent_at DATETIME")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


ensure_user_table_columns()
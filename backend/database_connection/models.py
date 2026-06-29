from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .database import Base

class Department(Base):
    __tablename__ = "departments"
    department_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    faqs = relationship("FAQ", back_populates="department")


class FAQ(Base):
    __tablename__ = "faqs"
    faq_id = Column(Integer, primary_key=True, index=True)
    department_id = Column(Integer, ForeignKey("departments.department_id"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_vector = Column(Vector(384))
    answer_vector = Column(Vector(384))
    enriched_vector = Column(Vector(384))

    department = relationship("Department", back_populates="faqs")


class Query(Base):
    __tablename__ = "queries"
    query_id = Column(Integer, primary_key=True, index=True)
    department_id = Column(Integer, ForeignKey("departments.department_id"))
    user_query = Column(Text, nullable=False)
    query_vector = Column(Vector(384))
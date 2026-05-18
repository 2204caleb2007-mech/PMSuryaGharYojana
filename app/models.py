"""
Database models — production schema
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum


class AuthProvider(str, enum.Enum):
    LOCAL = "local"
    GOOGLE = "google"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)   # nullable: Google users skip this
    hashed_password = Column(String(255), nullable=True)                     # nullable: Google users have no password
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    # ── Google OAuth fields ──────────────────────────────────────────────────
    google_id = Column(String(255), unique=True, index=True, nullable=True)  # Google subject (sub)
    profile_picture = Column(String(1000), nullable=True)                    # Google avatar URL
    auth_provider = Column(Enum(AuthProvider), default=AuthProvider.LOCAL, nullable=False)
    # ────────────────────────────────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    analyses = relationship("SolarAnalysis", back_populates="user", cascade="all, delete-orphan")
    subsidy_calculations = relationship("SubsidyCalculation", back_populates="user", cascade="all, delete-orphan")
    batch_jobs = relationship("CSVBatchJob", back_populates="user", cascade="all, delete-orphan")


class SolarAnalysis(Base):
    __tablename__ = "solar_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # null = ephemeral guest result
    sample_id = Column(String(100), unique=True, index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String(500), nullable=True)
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="pending")  # pending, completed, failed

    # Relationships
    user = relationship("User", back_populates="analyses")
    result = relationship("AnalysisResult", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    images = relationship("UploadedImage", back_populates="analysis", cascade="all, delete-orphan")
    subsidy_calculations = relationship("SubsidyCalculation", back_populates="analysis", cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("solar_analyses.id"), nullable=False, unique=True)
    has_solar = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    panel_count = Column(Integer, default=0)
    pv_area_sqm = Column(Float, nullable=True)
    capacity_kw = Column(Float, nullable=True)
    qc_status = Column(String(50), default="PENDING")
    model_used = Column(String(100), nullable=True)
    raw_predictions = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    analysis = relationship("SolarAnalysis", back_populates="result")


class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("solar_analyses.id"), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    source = Column(String(50), default="upload")  # upload, satellite
    satellite_provider = Column(String(50), nullable=True)  # esri, google
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    analysis = relationship("SolarAnalysis", back_populates="images")


class SubsidyCalculation(Base):
    __tablename__ = "subsidy_calculations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # null = guest (not saved)
    analysis_id = Column(Integer, ForeignKey("solar_analyses.id"), nullable=True)
    sample_id = Column(String(100), nullable=True)
    state = Column(String(100), nullable=False)
    estimated_capacity_kw = Column(Float, nullable=False)
    panel_count = Column(Integer, default=0)
    subsidy_min = Column(Float, nullable=False)
    subsidy_max = Column(Float, nullable=False)
    avg_subsidy = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="subsidy_calculations")
    analysis = relationship("SolarAnalysis", back_populates="subsidy_calculations")


class CSVBatchJob(Base):
    __tablename__ = "csv_batch_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=True)
    total_rows = Column(Integer, default=0)
    processed_rows = Column(Integer, default=0)
    failed_rows = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="batch_jobs")

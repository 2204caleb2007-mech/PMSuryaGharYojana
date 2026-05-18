"""
Subsidy calculation routes
POST /api/subsidy — calculate subsidy estimate
  Guest: returns calculation but does NOT persist
  Auth: returns AND stores in DB linked to user
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models import User, SubsidyCalculation, SolarAnalysis
from app.schemas import SubsidyRequest, SubsidyResponse
from app.dependencies import get_current_user_optional

router = APIRouter()

# ── State-wise subsidy table (INR) ──────────────────────────────────────────────
# Based on MNRE guidelines for PM Surya Ghar Yojana (2024)
SUBSIDY_TABLE: dict[str, dict] = {
    "Andhra Pradesh":        {"per_kw_min": 14588, "per_kw_max": 21892, "note": "State top-up available"},
    "Arunachal Pradesh":     {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Assam":                 {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Bihar":                 {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Chhattisgarh":          {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Delhi":                 {"per_kw_min": 20000, "per_kw_max": 30000, "note": "Delhi additional subsidy"},
    "Goa":                   {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Gujarat":               {"per_kw_min": 14588, "per_kw_max": 21892, "note": "State scheme top-up"},
    "Haryana":               {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Himachal Pradesh":      {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Jharkhand":             {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Karnataka":             {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Kerala":                {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Madhya Pradesh":        {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Maharashtra":           {"per_kw_min": 14588, "per_kw_max": 21892, "note": "MSEDCL net metering"},
    "Manipur":               {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Meghalaya":             {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Mizoram":               {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Nagaland":              {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Odisha":                {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Punjab":                {"per_kw_min": 14588, "per_kw_max": 21892, "note": "PSPCL net metering"},
    "Rajasthan":             {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Sikkim":                {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Tamil Nadu":            {"per_kw_min": 14588, "per_kw_max": 21892, "note": "TANGEDCO net metering"},
    "Telangana":             {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Tripura":               {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "Uttar Pradesh":         {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Uttarakhand":           {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Special category state"},
    "West Bengal":           {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Andaman and Nicobar Islands": {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Island territory"},
    "Chandigarh":            {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Dadra and Nagar Haveli and Daman and Diu": {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
    "Jammu and Kashmir":     {"per_kw_min": 14588, "per_kw_max": 21892, "note": "UT special provisions"},
    "Ladakh":                {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Remote area benefit"},
    "Lakshadweep":           {"per_kw_min": 14588, "per_kw_max": 21892, "note": "Island territory"},
    "Puducherry":            {"per_kw_min": 14588, "per_kw_max": 21892, "note": ""},
}

DEFAULT_RATES = {"per_kw_min": 14588, "per_kw_max": 21892}

# MNRE 2024 slab structure:
# ≤2 kW → ₹30,000 per kW (up to ₹60,000)
# 2-3 kW → ₹18,000 per kW for the additional (up to ₹78,000 total)
# >3 kW → capped at ₹78,000

def _calculate_mnre_subsidy(capacity_kw: float) -> dict:
    if capacity_kw <= 2:
        amount = capacity_kw * 30000
    elif capacity_kw <= 3:
        amount = 60000 + (capacity_kw - 2) * 18000
    else:
        amount = 78000

    state_rate = DEFAULT_RATES
    return {
        "min": round(amount * 0.85),
        "max": round(amount),
        "avg": round(amount * 0.925),
    }


@router.post("/", response_model=SubsidyResponse)
async def calculate_subsidy(
    body: SubsidyRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Calculate subsidy estimate.
    Guest: returns calculation without saving.
    Auth: persists linked to user (and optionally to a sample_id).
    """
    rates = SUBSIDY_TABLE.get(body.state, DEFAULT_RATES)
    mnre = _calculate_mnre_subsidy(body.estimated_capacity_kw)

    saved_to_db = False
    calc_id = None

    if current_user:
        # Optionally link to an analysis
        analysis_id = None
        if body.sample_id:
            analysis = db.query(SolarAnalysis).filter(
                SolarAnalysis.sample_id == body.sample_id,
                SolarAnalysis.user_id == current_user.id,
            ).first()
            if analysis:
                analysis_id = analysis.id

        db_calc = SubsidyCalculation(
            user_id=current_user.id,
            analysis_id=analysis_id,
            sample_id=body.sample_id,
            state=body.state,
            estimated_capacity_kw=body.estimated_capacity_kw,
            panel_count=body.panel_count or 0,
            subsidy_min=mnre["min"],
            subsidy_max=mnre["max"],
            avg_subsidy=mnre["avg"],
        )
        db.add(db_calc)
        db.commit()
        db.refresh(db_calc)
        saved_to_db = True
        calc_id = db_calc.id

    return SubsidyResponse(
        state=body.state,
        estimated_capacity_kw=body.estimated_capacity_kw,
        panel_count=body.panel_count or 0,
        subsidy_min=mnre["min"],
        subsidy_max=mnre["max"],
        avg_subsidy=mnre["avg"],
        currency="INR",
        sample_id=body.sample_id,
        saved_to_db=saved_to_db,
        calculation_id=calc_id,
    )

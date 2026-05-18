"""
Geocode proxy — hides GOOGLE_MAPS_API_KEY from frontend
Falls back to Nominatim (OpenStreetMap) if no Google key
GET /api/geocode?q=<address>
"""
import logging
import requests
from fastapi import APIRouter, Query, HTTPException
from app.core.config import settings
from app.schemas import GeocodeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=list[GeocodeResponse])
async def geocode(q: str = Query(..., min_length=3, description="Address or location query")):
    """
    Geocode a location string to lat/lon.
    Uses Google Maps if key is configured, otherwise Nominatim (OSM).
    """
    if settings.GOOGLE_MAPS_API_KEY:
        return await _google_geocode(q)
    return await _nominatim_geocode(q)


async def _google_geocode(q: str) -> list[GeocodeResponse]:
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": q, "key": settings.GOOGLE_MAPS_API_KEY}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", [])[:5]:
            loc = item["geometry"]["location"]
            results.append(GeocodeResponse(
                display_name=item["formatted_address"],
                latitude=loc["lat"],
                longitude=loc["lng"],
                address_components=item.get("address_components"),
            ))
        return results
    except Exception as e:
        logger.error(f"Google geocode error: {e}")
        raise HTTPException(status_code=503, detail="Geocoding service unavailable")


async def _nominatim_geocode(q: str) -> list[GeocodeResponse]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 5, "countrycodes": "in"}
        resp = requests.get(url, params=params, timeout=10, headers={
            "User-Agent": "SuryaSys-Solar-Analyzer/4.0 (contact@suryasys.gov.in)"
        })
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data:
            results.append(GeocodeResponse(
                display_name=item["display_name"],
                latitude=float(item["lat"]),
                longitude=float(item["lon"]),
            ))
        return results
    except Exception as e:
        logger.error(f"Nominatim geocode error: {e}")
        raise HTTPException(status_code=503, detail="Geocoding service unavailable")

"""
Satellite imagery services
"""
import requests
import math
import logging
from typing import Optional, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

class SatelliteImageryService:
    """Service for fetching satellite imagery"""
    
    @staticmethod
    def get_esri_tile_url(latitude: float, longitude: float, zoom: int = 18) -> str:
        """Get Esri World Imagery tile URL (free public service)"""
        def lat_lon_to_tile(lat, lon, z):
            n = 2 ** z
            x = int((lon + 180) / 360 * n)
            y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
            return x, y
        
        tile_x, tile_y = lat_lon_to_tile(latitude, longitude, zoom)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
        return url
    
    @staticmethod
    def get_google_static_image_url(latitude: float, longitude: float, zoom: int = 18, size: str = "640x640") -> Optional[str]:
        """Get Google Maps Static API URL (requires API key)"""
        if not settings.GOOGLE_MAPS_API_KEY:
            return None
        
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={latitude},{longitude}&"
            f"zoom={zoom}&"
            f"size={size}&"
            f"maptype=satellite&"
            f"key={settings.GOOGLE_MAPS_API_KEY}"
        )
        return url
    
    @staticmethod
    def fetch_satellite_image(latitude: float, longitude: float, zoom: int = 18, provider: str = "esri") -> Optional[bytes]:
        """
        Fetch satellite image for given coordinates
        Returns image bytes or None
        """
        try:
            if provider == "google" and settings.GOOGLE_MAPS_API_KEY:
                url = SatelliteImageryService.get_google_static_image_url(latitude, longitude, zoom)
            else:
                url = SatelliteImageryService.get_esri_tile_url(latitude, longitude, zoom)
            
            if not url:
                return None
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to fetch satellite image: {e}")
            return None


import geopandas as gpd
import pandas as pd
import random
import json
import os
from faker import Faker
from shapely.geometry import Point
from geopy.distance import geodesic
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any, Union, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with fallbacks
DEMOGRAPGIC_FILE = os.getenv('DEMOGRAPGIC_FILE',None)
POI_FILE = os.getenv('POI_FILE',None)
CENCUS_FILE = os.getenv('CENCUS_FILE',None)
BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434/v1')
API_KEY = os.getenv('API_KEY', '123')
MODEL = os.getenv('MODEL', 'qwen3')

class AgentProfile(BaseModel):
    name: str
    city: str
    age: int
    gender: str
    employment_status: str
    education: str
    household_size: int
    household_income: str
    available_vehicles: int

class POISearchTool:
    """A tool for searching Points of Interest (POIs) near a geographical location."""
    
    def __init__(self):
        """
        Initialize the POI search tool with a GeoDataFrame containing POI data.
        """
        poi_df = gpd.read_file(POI_FILE)
        if not isinstance(poi_df, gpd.GeoDataFrame):
            geometry = [Point(xy) for xy in zip(poi_df.LONGITUDE, poi_df.LATITUDE)]
            poi_df = gpd.GeoDataFrame(poi_df, geometry=geometry, crs="EPSG:4326")
        else:
            poi_df = poi_df.copy()
        # 创建空间索引
        poi_df.sindex
        poi_df['NAICS_CODE'] = poi_df['NAICS_CODE'].astype(int)
        self.poi_df = poi_df
        self.definition ={
            "type": "function",
            "function": {
                "name": "find_nearby_pois",
                "description": "Search for Points of Interest (POIs) of specific types near a geographical location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude of the current location"
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude of the current location"
                        },
                        "min_distance_km": {
                            "type": "number",
                            "description": "Minimum search radius in kilometers"
                        },
                        "max_distance_km": {
                            "type": "number",
                            "description": "Maximum search radius in kilometers"
                        },
                        "naics_code": {
                            "type": "string",
                            "description": "NAICS code(s) for the desired POI types (comma-separated for multiple)"
                        }
                    },
                    "required": ["lat", "lon", "min_distance_km", "max_distance_km", "naics_code"]
                }
            }
        }
    def find_nearby_pois(self, lat: float, lon: float, min_distance_km: float = 0, 
                         max_distance_km: float = None, naics_code: Union[str, List[int], int] = None) -> Dict:
        """
        Search for POIs near a geographical location with filtering options.
        
        Args:
            lat: Latitude of the current location
            lon: Longitude of the current location
            min_distance_km: Minimum search radius in kilometers
            max_distance_km: Maximum search radius in kilometers
            naics_code: NAICS code(s) for filtering POI types (can be string, list, or int)
            
        Returns:
            Dictionary containing search status and list of POIs
        """
        # Convert naics_code to proper format - could be string or list
        if isinstance(naics_code, str):
            # Check if it's a comma-separated list
            if ',' in naics_code:
                naics_list = [int(code.strip()) for code in naics_code.split(',')]
            else:
                naics_list = [int(naics_code.strip())]
        elif isinstance(naics_code, list):
            naics_list = [int(code) for code in naics_code]
        elif isinstance(naics_code, int):
            naics_list = [naics_code]
        else:
            naics_list = None

        # Create current point geometry
        current_point = Point(lon, lat)
        
        # Convert to UTM for accurate distance calculation
        utm_crs = self.poi_df.estimate_utm_crs()
        utm_gdf = self.poi_df.to_crs(utm_crs)
        current_point_utm = gpd.GeoSeries([current_point], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
        
        # Use spatial index for faster filtering
        buffer = current_point_utm.buffer(max_distance_km * 1000)
        candidates_idx = list(utm_gdf.sindex.intersection(buffer.bounds))
        candidates = utm_gdf.iloc[candidates_idx]
        in_buffer = candidates[candidates.intersects(buffer)].to_crs("EPSG:4326")
        
        # Calculate exact distances using geodesic
        in_buffer = in_buffer.copy()
        in_buffer['distance_km'] = [
            geodesic((lat, lon), (row['LATITUDE'], row['LONGITUDE'])).km 
            for _, row in in_buffer.iterrows()
        ]
        
        # Apply distance and NAICS filters
        result = in_buffer[(in_buffer['distance_km'] >= min_distance_km) & 
                        (in_buffer['distance_km'] <= max_distance_km)]
        
        if naics_list is not None:
            result = result[result['NAICS_CODE'].isin(naics_list)]

        results = pd.DataFrame(result.drop(columns='geometry', errors='ignore')
                      ).sort_values('distance_km').reset_index(drop=True)
        top_results = results.head(20).to_dict('records')
        
        formatted_results = []
        for poi in top_results:
            formatted_results.append({
                "name": poi['LOCATION_NAME'],
                "top_category": poi['TOP_CATEGORY'],
                "sub_category": poi['SUB_CATEGORY'],
                "distance_km": poi["distance_km"],
                "latitude": poi["LATITUDE"],
                "longitude": poi["LONGITUDE"]  # Fixed typo from original code
            })
        
        return {"status": "success", "pois": formatted_results}

    
class ProfileGeneratorTool:
    """Tool for generating realistic demographic profiles for mobility agents."""
    
    def __init__(self):
        self.client = OpenAI(base_url=BASE_URL,api_key=API_KEY)
        self.cbd_df = gpd.read_file(CENCUS_FILE)
        if DEMOGRAPGIC_FILE and os.path.exists(DEMOGRAPGIC_FILE):
            with open(DEMOGRAPGIC_FILE,'r') as f:
                self.demographic_data = json.load(f)
        self.definition = {
            "type": "function",
            "function": {
                "name": "generate_profile",
                "description": "Generate a realistic demographic profile for a mobility agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City where the person lives"
                        },
                        "age_bias": {
                            "type": "string",
                            "enum": ["younger", "older", "balanced"],
                            "description": "Optional bias to introduce in the age distribution"
                        },
                        "income_bias": {
                            "type": "string",
                            "enum": ["lower", "higher", "balanced"],
                            "description": "Optional bias to introduce in the income distribution"
                        }
                    },
                    "required": ["city"]
                }
            }
        }

    def generate_profile(self, city: str, age_bias: str = "balanced", income_bias: str = "balanced") -> AgentProfile:
        """
        Generate a realistic demographic profile for a mobility agent.
        
        Args:
            city: City where the person lives
            age_bias: Optional bias in age distribution ('younger', 'older', 'balanced')
            income_bias: Optional bias in income distribution ('lower', 'higher', 'balanced')
            
        Returns:
            A realistic demographic profile
        """
        def get_fakename(gender):
            fake = Faker()
            if gender == 'male':
                return fake.name_male()
            else:
                return fake.name_female()
        
        def weighted_sample(options, weights):
            return random.choices(options, weights=weights, k=1)[0]
        
        # Load demographic distributions from the stored data
        age_ranges = self.demographic_data["age_ranges"]
        age_weights = self.demographic_data["age_weights"]
        
        genders = self.demographic_data["genders"]
        gender_weights = self.demographic_data["gender_weights"]
        
        employment_statuses = self.demographic_data["employment_statuses"]
        employment_weights = self.demographic_data["employment_weights"]
        
        vehicles = self.demographic_data["vehicles"]
        vehicle_weights = self.demographic_data["vehicle_weights"]
        
        education_levels = self.demographic_data["education_levels"]
        education_weights = self.demographic_data["education_weights"]
        
        income_ranges = self.demographic_data["income_ranges"]
        income_weights = self.demographic_data["income_weights"]
        
        household_size = self.demographic_data["household_size"]
        household_weights = self.demographic_data["household_weights"]

        gender = weighted_sample(genders, gender_weights)

        sampled_profile = {
            "name": get_fakename(gender),
            "city": city,
            "age": weighted_sample(age_ranges, age_weights),
            "gender": gender,
            "employment_status": weighted_sample(employment_statuses, employment_weights),
            "education": weighted_sample(education_levels, education_weights),
            "household_income": weighted_sample(income_ranges, income_weights),
            "available_vehicles": weighted_sample(vehicles, vehicle_weights),
            "household_size": weighted_sample(household_size, household_weights)
        }

        # If no client is provided, just return the sampled profile
        if not self.client:
            if isinstance(sampled_profile["age"], tuple):
                sampled_profile["age"] = random.randint(sampled_profile["age"][0], sampled_profile["age"][1])
            return AgentProfile(**sampled_profile)
            
        # Create prompt for LLM to refine the profile
        system_prompt = """You are a demographic profile generator for an urban mobility simulation.
        Given basic sampled demographic data, correct it if needed and return a realistic profile with a name.
        Fix any unrealistic combinations (e.g., a 19-year-old with a PhD).
        Return ONLY a valid JSON object, no extra explanation."""
        
        user_prompt = f"""Given this sampled data for a resident of {city}, generate a corrected, coherent profile:
        {json.dumps(sampled_profile, indent=2)}
        Ensure the values are realistic and consistent.
        Return a single valid JSON object with these fields:
        - name
        - city
        - age
        - gender
        - employment_status
        - education
        - household_size
        - household_income
        - available_vehicles
        """

        try:
            # Call the LLM to refine the profile
            response = self.client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=AgentProfile,
                temperature=0.7,  # Add some variety
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in profile generation: {e}")
            # If LLM call fails, ensure age is a number not a tuple and return the sampled profile
            if isinstance(sampled_profile["age"], tuple):
                sampled_profile["age"] = random.randint(sampled_profile["age"][0], sampled_profile["age"][1])
            profile = AgentProfile(**sampled_profile)
            return profile.model_dump_json()
        
    def get_home(self, bgrp_id=None):
        """
        Get a random home location within a census block group.
        
        Args:
            bgrp_id: Census block group ID. If None, one will be randomly selected.
            
        Returns:
            Dictionary containing home location information with name and coordinates.
        """
        if not bgrp_id:
            bgrp_id = random.choices(self.cbd_df['customGeoId'], weights=self.cbd_df['peopleByHome'], k=1)[0]
        
        # Set random point in census block as home
        bgrp_geometry = self.cbd_df[self.cbd_df['customGeoId']==bgrp_id]['geometry']
        geometry = bgrp_geometry.iloc[0]
        
        # Generate a random point within the polygon
        minx, miny, maxx, maxy = geometry.bounds
        import time
        start_time = time.time()
        
        while True:
            # Generate a random point within the bounding box
            random_point = Point(
                random.uniform(minx, maxx),
                random.uniform(miny, maxy)
            )
            
            # Check if the point is within the polygon
            if random_point.within(geometry):
                # Return as (latitude, longitude)
                return {
                    "name": "home",
                    'coordinates': (random_point.y, random_point.x)
                }
            elif time.time()-start_time > 1000:
                point = geometry.centroid
                return {
                    "type": "home",
                    "name": "my home",
                    'coordinates': (point.y, point.x)
                }
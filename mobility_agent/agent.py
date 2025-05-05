
from openai import OpenAI
import os
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from geopy.distance import geodesic
import random
import datetime

from .graph import BehaviorGraph
from .tools import POISearchTool, ProfileGeneratorTool

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with fallbacks
REFERENCE_CITY = os.getenv('REFERENCE_CITY', 'Cambridge,MA')
BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434/v1')
API_KEY = os.getenv('API_KEY', '123')
MODEL = os.getenv('MODEL', 'qwen3')


class ChoiceWeight(BaseModel):
    primary_mode: str
    duration_minutes: str
    weight: float

class TransportationChoice(BaseModel):
    think: List[str] = Field(description="List of reasons for your answer (up to 4 items, no more than 20 words for each item)")
    choice_weights: List[ChoiceWeight]

class ScheduleItem(BaseModel):
    start_time: str = Field(description="Start time in HH:MM format (24-hour clock)")
    desire: str = Field(description="Type of activity, one of ['school', 'home','shop', 'recreation', 'eat', 'social','maintenance', 'work','other_activity_type']")

class DailySchedule(BaseModel):
    schedule: List[ScheduleItem] = Field(description="List of schedule items for the day")

class POISelection(BaseModel):
    reasoning: str = Field(description="Reasoning for the selection in a short sentence (no more than 20 words)")
    selection: int = Field(description="The selected destination option number")
    
class MobilityAgent:
    def __init__(self,profile=None,city='Cambridge,MA',sample_num=1000,reference_city=REFERENCE_CITY):
        self.client = OpenAI(base_url=BASE_URL,api_key=API_KEY)
        self.profile = profile
        self.city = city
        self.tools={
            "poi_search": POISearchTool(),
            "profile_generator": ProfileGeneratorTool()
        }
        self.reference_city = reference_city
        self.behavior_graph = BehaviorGraph(sample_num=sample_num)
        self.memory_locations = {}
        self.working_memory = ["Today is a normal weekday"]
        if not profile:
            self._generate_profile()
        return
    
    def _generate_profile(self):
        self.profile = self.tools['profile_generator'].generate_profile(city=self.city)
        home = self.tools['profile_generator'].get_home()
        self._update_memory_location(location_type="home",location_name=home['name'],coordinates=home['coordinates'])
        return
    
    def _get_nearby_poi(self, current_lat, current_lon, min_distance_km=0, max_distance_km=None, naics_code=None):
        return self.tools['poi_search'].find_nearby_pois(
            lat=current_lat,
            lon=current_lon,
            min_distance_km=min_distance_km,
            max_distance_km=max_distance_km,
            naics_code=naics_code
        )
    
    def _update_memory_location(self, location_type, location_name, coordinates,additional_info=None):
        """Update agent memory with important locations."""
        location_data = {
            "name": location_name,
            "coordinates": coordinates
        }
        if additional_info:
            location_data.update(additional_info)
        # Also update the working memory string to include this information
        self.memory_locations[location_type] = location_data
        return location_data
        
    def get_mode_prefernce(self, desire,time,distance=None):
        _,weights =self.behavior_graph.preference_modelling(profile=self.profile,desire=desire,time=time)

        if not distance : 
            distance = 'unknown'

        choice_template = f""" You are {self.profile}, tasked with simulating realistic transportation behavior based on reference data from people with similar profiles. 

        Current Time: {time}
        Current Desire: {desire}
        Current memory: {self.working_memory}
        Context: This is the reference choices of people who have similar profiles in {self.reference_city} in a normal workday. Higher wights means more likely to be chosen by people with similar profile.
        Target Distance: {distance}

        What transportation mode and duration minutes do you prefer?

        Available Mode Choices : {[d['primary_mode'] for d in weights]}
        Available Duration Choices : {[d['duration_minutes'] for d in weights]}

        Note: 
        - Output your reason and preference weight of different choices
        - You should consider your profile, the choices, and the possible cultural difference between {self.reference_city} and {self.city}.
        - The sum of all weights should equals to 1.

        Answer Format:
        {{
        'think': [list your reason in short sentences (up to 4 points)]
        'choice_weights':[
        {{'primary_mode': mode1, 'duration_minutes': duration1, weight: weight1 }},
        {{'primary_mode': mode2, 'duration_minutes': duration2, weight: weight2 }},
        ...
        ]
        }}
        """
        system_prompt = "You are a transportation behavior simulator that makes realistic choices based on statistical patterns."

        try:
            response = self.client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": choice_template}
            ],
            temperature=0.7,  # Some randomness for variety
            response_format=TransportationChoice,
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(e)
            return None
    
    def get_location_choice(self,current_lat,current_lon,desire,time):
        # First check if this destination type exists in memory
        if desire in self.memory_locations:
            # Use the existing location from memory
            memorized_location = self.memory_locations[desire]
            
            # Calculate distance to determine updated travel duration
            destination_coords = memorized_location['coordinates']
            current_coords = (current_lat, current_lon)
            distance_km = geodesic(current_coords, destination_coords).kilometers
            
            # Get mode preferences still to determine travel mode
            mode_preference = self.get_mode_prefernce(desire=desire, time=time, distance=distance_km)
            choice_weights = mode_preference['choice_weights']

            # Select the mode with the highest weight
            selected_mode_idx = max(range(len(choice_weights)), key=lambda i: choice_weights[i]['weight'])
            selected_mode = choice_weights[selected_mode_idx]

            # Return the result with the memorized location
            return {
                'agent_profile': self.profile,
                'desire': desire,
                'time': time,
                'transportation': {
                    'mode': selected_mode['primary_mode'],
                    'duration': selected_mode['duration_minutes']
                },
                'destination': {
                    'name': memorized_location['name'],
                    'distance_km': distance_km,
                    'coordinates': memorized_location['coordinates']
                },
                'reasoning': f"This is my regular {desire} location."
            }
        
        mode_prefernce = self.get_mode_prefernce(desire=desire,time=time)
        choice_weights = mode_prefernce['choice_weights']
        modes = [choice['primary_mode'] for choice in choice_weights]
        weights = [choice['weight'] for choice in choice_weights]
        selected_mode_idx = random.choices(range(len(modes)), weights=weights, k=1)[0]
        selected_mode = choice_weights[selected_mode_idx]

        system_prompt = f"""You are a mobility planning assistant that helps find appropriate destinations.
        You have access to a tool that can search for Points of Interest (POIs) near a location.
        
        Call the find_nearby_pois tool with appropriate parameters based on the transportation mode, travel time, and activity desired.

        When determining search parameters, consider:
        - For walking, a typical speed is ~5 km/h
        - For biking, a typical speed is ~15 km/h
        - For driving, a typical urban speed is ~30 km/h
        - For public transit, a typical speed is ~20 km/h

        NAICS codes for common destinations:
        - Shopping: 445110 (grocery stores), 448140 (clothing stores), 443142 (electronics stores), 452210 (department stores)
        - Food/Restaurants: 722511 (full-service restaurants), 722513 (fast food), 722515 (coffee/snack bars), 722410 (bars/drinking places)
        - Education/School: 611110 (K-12 schools), 611310 (colleges/universities), 611610 (fine arts schools)
        - Recreation/Entertainment: 713940 (gyms/fitness centers), 712130 (zoos/gardens), 512131 (movie theaters), 713950 (bowling centers)
        - Social/Community: 813410 (civic/social orgs), 624110 (youth services), 624120 (elderly/disability services), 813910 (business associations), 531110 (apartment rentals), 531311 (property managers), 236118 (residential remodelers), 621610 (home health care)
        - Maintenance/Repair: 811111 (auto repair), 811192 (car washes), 561730 (landscaping), 811411 (home equipment repair)
        - Other/Public Services: 813110 (religious organizations), 921110 (government offices), 922120 (police), 922160 (fire dept),541110 (law offices), 621111 (physician offices), 541512 (computer systems design), 541611 (management consulting)
        """
        
        user_prompt = f"""I am {self.profile} at {time}:00, wanting to {desire}. 
        I'll be traveling by {selected_mode['primary_mode']} with an estimated travel time of {selected_mode['duration_minutes']} minutes.

        Memory: {self.working_memory}
        
        Please help me find suitable destinations by:
        1. Check if the corresponding location exists in memory (e.g., "My Home", "My Office", "My School"). If found, use that saved location and skip steps 2-5, otherwise continue.
        2. Determining an appropriate min and max search radius based on my transportation mode and travel time
        3. Identifying the most relevant NAICS codes for my desired activity and time of day
        4. Calling the find_nearby_pois tool to search for destinations.
        5. Recommending the best option from the results
        
        My current location is at latitude {current_lat}, longitude {current_lon}.
        """

        tools = [self.tools['poi_search'].definition]

        # Call the LLM with tool calling capability
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            tool_choice="auto",
        )
        
        message = response.choices[0].message
        # Handle the tool calls
        if message.tool_calls:
            # Get the tool call
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            tool_result = self._get_nearby_poi(
                function_args.get("lat"),
                function_args.get("lon"),
                function_args.get("min_distance_km"),
                function_args.get("max_distance_km"),
                function_args.get("naics_code")
            )
            
            # Get LLM to choose the best POI from the results
            pois_found = len(tool_result["pois"])
            # print(f"Found {pois_found} matching POIs nearby.")
            
            if pois_found == 0:
                # print("No suitable destinations found.")
                return None
            
            # Prepare POI descriptions for the LLM
            poi_text = json.dumps(tool_result["pois"])
            
            # LLM selection of the best POI
            user_prompt_selection = f"""Based on my profile ({self.profile}), time ({time}:00), 
            transportation mode ({selected_mode['primary_mode']}), prefered duration ({selected_mode['duration_minutes']} minutes), and desire ({desire}), 
            which of these locations would be the most suitable?
            
            Available destinations:
            {poi_text}
            
            Please select the number of the best option (1-{len(tool_result["pois"])}) and explain your reasoning.
            """
            try:
                response_selection = self.client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a personal assistant helping to select the most appropriate destination."},
                        {"role": "user", "content": user_prompt_selection}
                    ],
                    response_format=POISelection,
                )
                selection_response = json.loads(response_selection.choices[0].message.content)
                selection_idx = int(selection_response.get("selection", 1)) - 1
                selection_text = selection_response.get("reasoning", "")
            except Exception as e:
                print(f"Error parsing location selection: {e}")
                selection_idx = 0
                selection_text = "Default selection due to parsing error"

            # Ensure index is valid
            if selection_idx < 0 or selection_idx >= len(tool_result["pois"]):
                selection_idx = 0
            
            # Get the selected POI
            selected_poi = tool_result["pois"][selection_idx]

            # Format the result
            result = {
                'agent_profile': self.profile,
                'desire': desire,
                'time': time,
                'transportation': {
                    'mode': selected_mode['primary_mode'],
                    'duration': selected_mode['duration_minutes']
                },
                'destination': {
                    'name': selected_poi['name'],
                    'distance_km': selected_poi['distance_km'],
                    'coordinates': (selected_poi['latitude'], selected_poi['longitude'])
                },
                'reasoning': selection_text
            }

            # Update memory for important locations (work, school, home)
            if desire == "work" and "work" not in self.memory_locations:
                self._update_memory_location(
                    location_type="work",
                    location_name=selected_poi['name'],
                    coordinates=(selected_poi['latitude'], selected_poi['longitude']),
                    additional_info={"category": selected_poi.get('category', '')}
                )
            elif desire == "school" and "school" not in self.memory_locations:
                self._update_memory_location(
                    location_type="school",
                    location_name=selected_poi['name'],
                    coordinates=(selected_poi['latitude'], selected_poi['longitude']),
                    additional_info={"category": selected_poi.get('category', '')}
                )
            
            return result
        else:
            # Handle case where no tool was called
            # print("The LLM didn't call the search tool as expected.")
            return None
        
    def make_a_plan(self):
        subgraph,_ =self.behavior_graph.preference_modelling(profile=self.profile,desire='work',time=10)

        reference_schedule = [d['props'] for _,d in subgraph.nodes(data=True) if d['type']=='desire']

        system_prompt = """You are an expert in human mobility patterns and daily routines.
        Your task is to create a realistic daily transport schedule for a person based on their profile and the memory."""
        user_prompt = f"""Create a detailed daily schedule for {self.profile}. 
        
        Current memory: {self.working_memory}

        IMPORTANT: Only include entries when the person needs to **travel to a new location**. 
        Do NOT include activities that happen at the current location (e.g., eating breakfast at home while already at home should not be included).

        This is some reference schedule from people with similar profile : {reference_schedule}
        
        The schedule should include:
        1. Start time for each activity in HH:MM format (24-hour clock)
        2. The type of activity from these categories: school, home, shop, recreation, eat, social, maintenance, work, other_activities
        
        Return the schedule as a structured object with a list of activities, each having a start time and desire.
        Make sure the schedule represents a complete day and is realistic.
        """
        # Call the LLM
        try:
            response = self.client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=DailySchedule,
            )
            plan = json.loads(response.choices[0].message.content)
            return plan
        except Exception as e:
            # print(f"Error generating schedule: {e}")
            return None
        
    def get_time_schedule(self):
        def strtime2number(strtime):
            time = datetime.datetime.strptime(strtime, "%H:%M").time()
            return time.hour+time.minute/60
        
        # Try to load existing schedule
        load_existing_schedule = self.load_schedule()
        if load_existing_schedule:
            return self.time_schedule
        
        # If we couldn't load, generate a new schedule
        if "home" not in self.memory_locations:
            # print("Home location not found in memory. Generate profile first.")
            self._generate_profile()
        prev_location = self.memory_locations["home"]["coordinates"]
        prev_desire = "home"

        plan = self.make_a_plan()
        self.time_schedule = []

        if plan:
            for p in plan['schedule']:
                time = p['start_time']
                desire = p['desire']
                if desire != prev_desire:
                    try:
                        result = self.get_location_choice(current_lat=prev_location[0],current_lon=prev_location[1],desire=desire,time=strtime2number(time))
                        if result:
                            schedule_item = {
                                "start_time":time,
                                "desire":desire,
                                'transportation':result['transportation'],
                                "destination":result['destination'],
                            }
                            self.time_schedule.append(schedule_item)
                            self.working_memory.append(f"On {time}, I want to {desire}, I go to {result['destination']['name']} by {result['transportation']['mode']} which takes {result['transportation']['duration']} minutes.")
                    except Exception as e:
                        print(e)
                prev_desire = desire

        if len(self.time_schedule):
            self.save_schedule()
        return self.time_schedule
    
    def save_schedule(self):
        """Save the current time schedule and agent state to a JSON file."""
        # Create a directory for the schedules if it doesn't exist
        os.makedirs("data/agents", exist_ok=True)
        
        # Generate a filename based on the agent's profile
        profile_id = hash(self.profile)
        filename = f"data/agents/agent_{profile_id}.json"
        
        # Save the schedule and agent state
        agent_data = {
            "profile": self.profile,
            "city": self.city,
            "reference_city": self.reference_city,
            "schedule": self.time_schedule,
            "memory_locations": self.memory_locations,
            "working_memory": self.working_memory,
        }
        
        with open(filename, "w") as f:
            json.dump(agent_data, f, indent=2)
        
        # print(f"Agent state and schedule saved to {filename}")
        return filename
    
    def load_schedule(self):
        """Load a previously saved schedule and agent state if available."""
        # Generate the expected filename
        profile_id = hash(self.profile)
        filename = f"data/agents/agent_{profile_id}.json"
        
        # Check if the file exists
        if not os.path.exists(filename):
            return False
        
        with open(filename, "r") as f:
            data = json.load(f)
        
        # Restore the agent state
        self.time_schedule = data.get("schedule", [])
        self.memory_locations = data.get("memory_locations", {})
        self.working_memory = data.get("working_memory", ["Today is a normal weekday"])
        
        # Optionally restore other fields if needed
        if "city" in data and data["city"] != self.city:
            print(f"Note: Loaded agent was from {data['city']}, current city is {self.city}")
        
        # print(f"Loaded agent state and schedule from {filename}")
        return True


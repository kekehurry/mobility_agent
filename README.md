
# Init Environment

## Create and activate a virtual environment
```
uv venv
source .venv/bin/activate
```

## Install dependencies from pyproject.toml
```
uv sync
```

## Download POI Data

```
gdown 1ZMA_tNP6EapE6UED8hyi5_6uoNzTSvt0 -O data/geo/safegraph-sf-poi.csv
gdown 1JnOaM22HSrWC8SWVKm2x7kJHjl0oHoVt -O data/geo/safegraph-cambridge-poi.csv
```

## Create Qwen3-nothink Model

The experiments are using `qwen3:8b`, and the thinking mode is off by default. Using the `Modelfile` to create this model

```
ollama create qwen3-nothink:8b -f Modelfile
```

# Example Usage

## Mode Preference Modelling

```python
from mobility_agent.graph import BehaviorGraph
graph = BehaviorGraph()

profile = "a young women"
desire = "eat"
time = 8
subgraph,weights = graph.preference_modelling(profile=profile,desire=desire,time=time,k=10,depth=4)
graph.visualize_graph(subgraph,node_size=600,font_size=16,title_size=16)
print("Transportation Mode Preference:",weights)
```

## Agent-based Modelling

### Generate Profile

- The agent generate realistic profiles based on demongraphic data
- It then choose a random location as home wighted by cencus block population

```python
from mobility_agent.agent import MobilityAgent

agent = MobilityAgent(city="Cambridge,MA")
print("Profile:",agent.profile)
print("Home:", agent.memory_locations)
```

### Create Time Schedule

- The agent first creates a full-day activity plan.

- For each activity, it searches for nearby points of interest (POIs) based on their profile, and prefered mode of transportation.

- It then selects the most suitable POI to visit and save it to the working meomory of the agent.

- The agent will save important locations such as home, work, school to `agent.memory_locations`

```python
agent.get_time_schedule()
print("Time Schudule:",agent.time_schedule)
```

### Memories


```
print(agent.working_memory)
```


```
print(agent.memory_locations)
```




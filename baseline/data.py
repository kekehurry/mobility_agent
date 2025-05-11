from sklearn.preprocessing import StandardScaler
import datetime
import math
import pandas as pd


def number2strtime(numeric_time):
    hours = int(numeric_time)
    minutes = int((numeric_time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def strtime2number(strtime):
    time = datetime.datetime.strptime(strtime, "%H:%M:%S").time()
    return time.hour

def durationt2str(duration_time):
    min_duration = int(math.floor(duration_time/10)*10)
    max_duration = int(math.ceil(duration_time/10)*10)
    if min_duration == max_duration:
        max_duration += 10
    return f"{min_duration}-{max_duration}"

def categorize_age(age):
    """Categorize age into meaningful groups"""
    if age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

def categorize_income(income):
    """Categorize numeric household income into more evenly distributed brackets"""
    try:
        income_value = float(income)
        if income_value < 10000:
            return "Under $10k"
        if income_value < 50000:
            return "$10k-$50k"
        elif income_value < 100000:
            return "$50k-$100k"
        elif income_value < 150000:
            return "$100k-$150k"
        elif income_value < 200000:
            return "$150k-$200k"
        elif income_value < 300000:
            return "$200k-$300k"
        else:
            return "$300k+"
    except (ValueError, TypeError):
        # If income is not a valid number
        return "Unknown"
    

def load_data(trip_file):

    trip_df = pd.read_csv(trip_file)

    trip_df = trip_df.rename(columns={
        'trip_purpose': 'trip_purpose',
        'trip_start_time':'start_time',
        'primary_mode':'primary_mode',
        'trip_duration_minutes':'duration_minutes',
        'destination_land_use':'target_landuse',
        'trip_taker_person_id':'person_id',
        'trip_taker_household_id':'household_id',
        'trip_taker_age': 'age',
        'trip_taker_sex':'gender',
        'previous_trip_purpose':'previous_status',
        'trip_taker_employment_status':'employment_status',
        'trip_taker_household_size':'household_size',
        'trip_taker_household_income':'household_income',
        'trip_taker_available_vehicles':'available_vehicles',
        'trip_taker_industry':'industry',
        'trip_taker_education':'education',
        'trip_taker_work_bgrp_2020':'work_bgrp',
        'trip_taker_home_bgrp_2020':'home_bgrp'
    })

    trip_df['duration_minutes'] = trip_df['duration_minutes'].apply(durationt2str)
    trip_df['start_time'] = trip_df['start_time'].apply(strtime2number)
    trip_df['age_group'] = trip_df['age'].apply(categorize_age)
    trip_df['income_group'] = trip_df['household_income'].apply(categorize_income)

    trip_df = trip_df[['person_id','age','gender','employment_status','household_size','household_income','available_vehicles','industry','education','trip_purpose','start_time','primary_mode','duration_minutes','age_group','income_group']]
    
    return trip_df


def prepare_data(trip_df, encoder=None):
    # Define input and output features
    X_props = ['age_group','income_group', 'employment_status', 'household_size','available_vehicles', 'education', 'trip_purpose', 'start_time']
    y_props = ['primary_mode', 'duration_minutes']

    label_dict = {
    'age_group': ['Under 18', '18-24','25-34', '35-44','45-54','55-64', '65+'],'income_group': ['Under $10k', '$10k-$50k', '$50k-$100k','$100k-$150k','$150k-$200k', '$200k-$300k', '$300k+'], 
    'employment_status': ['under_16','not_in_labor_force', 'unemployed','employed',],
    'household_size': [str(i) for i in range(1,9)],
    'available_vehicles': ['zero','one','two','three_plus','unknown_num_vehicles'],
    'education': ['no_school','k_12','high_school','bachelors_degree', 'advanced_degree', 'some_college'], 
    'trip_purpose': ['eat', 'work', 'home', 'school', 'shop','maintenance', 'social', 'recreation','other_activity_type'], 
    'start_time': [str(i) for i in range(24)],
    'primary_mode': ['walking', 'biking', 'auto_passenger', 'public_transit','private_auto',  'on_demand_auto','other_travel_mode'], 
    'duration_minutes': ['0-10','10-20', '20-30', '30-40', '40-50', '50-60']
    }

    synthetic_rows = []
    for cat_feature in y_props:
        labels = set(label_dict[cat_feature])
        unique_values = set(trip_df[cat_feature].astype('str').unique())
        missing_categories = labels - unique_values
        for cat in missing_categories:
            sythetic_row = trip_df.iloc[0].copy()
            sythetic_row[cat_feature] = cat
            synthetic_rows.append(sythetic_row)

    if len(synthetic_rows)>0:
        trip_df = pd.concat([trip_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        
    # Split data into features and targets before preprocessing
    X_df = trip_df[X_props]
    y_df = trip_df[y_props]
    
    # Create label encoders for categorical features in X
    X_encoded = X_df.copy()
    y_encoded = y_df.copy()

    for cat_feature in X_props:
        labels = label_dict[cat_feature]
        mapping = {label: idx for idx, label in enumerate(labels)}
        X_encoded[cat_feature] = X_df[cat_feature].astype(str).map(mapping)
        # Replace NaN values with a default value (-1 or most common)
        if X_encoded[cat_feature].isna().any():
            X_encoded[cat_feature] = X_encoded[cat_feature].fillna(-1)

    for cat_feature in y_props:
        labels = label_dict[cat_feature]
        mapping = {label: idx for idx, label in enumerate(labels)}
        y_encoded[cat_feature] = y_df[cat_feature].astype(str).map(mapping)
        if y_encoded[cat_feature].isna().any():
            y_encoded[cat_feature] = y_encoded[cat_feature].fillna(-1)

    return X_encoded, y_encoded, encoder
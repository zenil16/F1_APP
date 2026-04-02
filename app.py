import streamlit as st
import google.generativeai as genai
import dotenv
import os
import pandas as pd
import fastf1
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import time
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# --- Setup ---
if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Page Configuration ---
st.set_page_config(
    page_title="🏎️ F1Expert AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Styling ---
def get_base64_of_image(image_url):
    with open(image_url, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_file):
    """Set the background image of the app."""
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        # If the image doesn't exist, use a CSS gradient
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e1e1e, #2d2d2d, #1e1e1e);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            @keyframes gradient {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Try to set the background image (if file exists)
try:
    set_background("f1_background.jpg")
except:
    # Use CSS gradient background if image doesn't exist
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e1e, #2d2d2d, #1e1e1e);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
        }
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        h1, h2, h3, h4 {
            color: #f2f2f2;
            font-family: 'Racing Sans One', 'Helvetica', sans-serif;
        }
        .stButton>button {
            background-color: #e10600;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #b30500;
            transform: scale(1.05);
        }
        .chat-container {
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            max-height: 500px;
            overflow-y: auto;
        }
        .user-message {
            background-color: rgba(14, 17, 23, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 15px 15px 0 15px;
            margin: 5px 0;
        }
        .bot-message {
            background-color: rgba(225, 6, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 15px 15px 15px 0;
            margin: 5px 0;
        }
        .prediction-container {
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
        }
        .trivia-question {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #e10600;
        }
        .trivia-answer {
            background-color: rgba(0, 100, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .live-tracking {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 10px;
            margin-top: 10px;
            color: white;
        }
        .driver-card {
            background: linear-gradient(145deg, rgba(30,30,30,0.8), rgba(60,60,60,0.8));
            border-radius: 8px;
            padding: 8px;
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.2s;
        }
        .driver-card:hover {
            transform: translateX(5px);
        }
        .position {
            font-weight: bold;
            font-size: 1.2rem;
            color: #e10600;
        }
        .driver-name {
            font-weight: bold;
        }
        .driver-time {
            color: #cccccc;
            font-family: monospace;
        }
        table {
            color: white !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
        }
        .dataframe {
            color: white !important;
        }
        .st-emotion-cache-1y4p8pa {
            border: 1px solid #333 !important;
            border-radius: 12px !important;
            background-color: rgba(0, 0, 0, 0.7) !important;
            color: white !important;
        }
        .st-emotion-cache-1y4p8pa th {
            background-color: rgba(225, 6, 0, 0.8) !important;
            color: white !important;
        }
        .st-emotion-cache-1y4p8pa td {
            background-color: rgba(30, 30, 30, 0.7) !important;
            color: white !important;
        }
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(60, 60, 60, 0.5);
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(225, 6, 0, 0.8) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Gemini Response Function ---
def response(messages):
    try:
        response = model.generate_content(messages)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# --- Chat History ---
def fetch_conversation_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "user", "parts": "System prompt: You are an F1 knowledge Expert - You have all the Knowledge about F1 and their Drivers - You are an intelligent Formula 1 assistant named RaceMaster."}
        ]
    return st.session_state["messages"]

# --- Trivia Question Generator ---
def generate_f1_trivia():
    """Generate 5 clean Formula 1 trivia questions and answers."""
    trivia_prompt = """
    Generate exactly 5 unique Formula 1 trivia questions and very short, direct answers.
    Answer MUST be very short (driver name, team, year, number, etc.) without explanation.
    Format the output as JSON inside ```json ``` block, like:

    ```json
    [
        {"question": "Who won the 2021 F1 World Championship?", "answer": "Max Verstappen"},
        {"question": "Which team has the most Constructors' Championships?", "answer": "Ferrari"}
    ]
    ```
    Only include short, direct answers. No explanations, no extra sentences.
    """

    try:
        response = model.generate_content(trivia_prompt)

        import json
        import re

        # Extract JSON part safely
        text = response.text
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # If no proper ```json ``` block found, treat full text as JSON
            json_str = text

        json_str = json_str.replace('\\n', '').replace('\\', '')

        # Parse JSON
        questions = json.loads(json_str)

        # --- Clean the answers to be SHORT only ---
        for qa in questions:
            if 'answer' in qa:
                # Remove if there are sentences or too much text
                qa['answer'] = qa['answer'].split('.')[0].strip()

        return questions

    except Exception as e:
        st.error(f"Trivia generation error: {str(e)}")

        # --- Safe fallback trivia ---
        return [
            {"question": "Who holds the most F1 World Championships?", "answer": "Lewis Hamilton"},
            {"question": "Which team is nicknamed the Silver Arrows?", "answer": "Mercedes"},
            {"question": "Which country hosts the Monaco Grand Prix?", "answer": "Monaco"},
            {"question": "Who was Ferrari's first F1 world champion?", "answer": "Alberto Ascari"},
            {"question": "What tire supplier is used in Formula 1?", "answer": "Pirelli"}
        ]


# --- Get Latest Year ---
@st.cache_data(show_spinner=False)
def get_latest_available_year():
    current_year = datetime.now().year
    for year in range(current_year, 2017, -1):
        try:
            schedule = fastf1.get_event_schedule(year)
            if schedule is not None and not schedule.empty:
                return year
        except Exception:
            continue
    return 2024

@st.cache_data(show_spinner=False)
def get_predictable_years():
    latest_year = get_latest_available_year()
    valid_years = []

    for year in range(2018, latest_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
            if schedule is None or schedule.empty:
                continue

            valid_schedule = schedule[
                (schedule['EventFormat'].notna()) &
                (~schedule['EventName'].astype(str).str.contains("Testing|Test", case=False, na=False))
            ].copy()

            if not valid_schedule.empty:
                valid_years.append(year)
        except Exception:
            continue

    return valid_years


@st.cache_data(show_spinner=False)
def get_valid_schedule_for_year(year):
    schedule = fastf1.get_event_schedule(year)

    valid_schedule = schedule[
        (schedule['EventFormat'].notna()) &
        (~schedule['EventName'].astype(str).str.contains("Testing|Test", case=False, na=False))
    ].copy()

    return valid_schedule


def resolve_historical_event_name(year, target_gp_name, target_location=None):
    try:
        schedule = get_valid_schedule_for_year(year)

        if schedule.empty:
            return None

        # 1. Exact match
        exact_match = schedule[schedule['EventName'] == target_gp_name]
        if not exact_match.empty:
            return exact_match.iloc[0]['EventName']

        # 2. Match by location
        if target_location:
            loc_match = schedule[schedule['Location'] == target_location]
            if not loc_match.empty:
                return loc_match.iloc[0]['EventName']

        # 3. Partial contains match
        partial = schedule[
            schedule['EventName'].astype(str).str.contains(target_gp_name.split()[0], case=False, na=False)
        ]
        if not partial.empty:
            return partial.iloc[0]['EventName']

    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False)
def get_predictable_gps(year):
    try:
        schedule = get_valid_schedule_for_year(year)
        if schedule.empty:
            return [], {}

        races = []
        locations = {}

        for _, row in schedule.iterrows():
            gp_name = row['EventName']
            gp_location = row['Location']

            historical_years = list(range(max(2018, year - 3), year))
            success_count = 0

            for hist_year in historical_years:
                resolved_name = resolve_historical_event_name(hist_year, gp_name, gp_location)
                if not resolved_name:
                    continue

                try:
                    session = fastf1.get_session(hist_year, resolved_name, 'R')
                    session.load()
                    if session.results is not None and not session.results.empty:
                        success_count += 1
                except Exception:
                    continue

            # Keep only races with at least 1 usable historical season
            if success_count >= 1:
                races.append(gp_name)
                locations[gp_name] = gp_location

        return races, locations

    except Exception:
        return [], {}

# --- Prediction Function ---
def predict_all_positions(gp_name, upcoming_year=None, gp_location=None):
    if upcoming_year is None:
        upcoming_year = get_latest_available_year()

    historical_years = list(range(max(2018, upcoming_year - 3), upcoming_year))
    all_races = []

    for year in historical_years:
        resolved_name = resolve_historical_event_name(year, gp_name, gp_location)
        if not resolved_name:
            continue

        try:
            session = fastf1.get_session(year, resolved_name, 'R')
            session.load()

            if session.results is None or session.results.empty:
                continue

            results = session.results[['Abbreviation', 'Position', 'TeamName']].copy()
            results['Year'] = year
            results = results.dropna(subset=['Abbreviation', 'Position', 'TeamName'])

            if not results.empty:
                all_races.append(results)

        except Exception as e:
            print(f"Skipping {gp_name} ({resolved_name}) in {year}: {e}")
            continue

    if not all_races:
        return None, None, None

    df = pd.concat(all_races, ignore_index=True)

    if df.empty:
        return None, None, None

    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])

    if df.empty:
        return None, None, None

    try:
        resolved_upcoming_name = resolve_historical_event_name(upcoming_year, gp_name, gp_location) or gp_name

        upcoming_session = fastf1.get_session(upcoming_year, resolved_upcoming_name, 'R')
        upcoming_session.load()

        if upcoming_session.results is not None and not upcoming_session.results.empty:
            upcoming_drivers = upcoming_session.results[['Abbreviation', 'TeamName']].drop_duplicates().copy()
            actual_results = upcoming_session.results[['Abbreviation', 'Position']].copy()
            show_actual = True
        else:
            raise ValueError("No results in upcoming session")

    except Exception:
        latest_year = int(df['Year'].max())
        upcoming_drivers = df[df['Year'] == latest_year][['Abbreviation', 'TeamName']].drop_duplicates().copy()
        actual_results = None
        show_actual = False

    if upcoming_drivers.empty:
        return None, None, None

    le_driver = LabelEncoder()
    le_driver.fit(list(pd.concat([df['Abbreviation'], upcoming_drivers['Abbreviation']]).astype(str).unique()))

    le_team = LabelEncoder()
    le_team.fit(list(pd.concat([df['TeamName'], upcoming_drivers['TeamName']]).astype(str).unique()))

    df['Driver_encoded'] = le_driver.transform(df['Abbreviation'].astype(str))
    df['Team_encoded'] = le_team.transform(df['TeamName'].astype(str))

    X = df[['Driver_encoded', 'Team_encoded', 'Year']]
    y = df['Position']

    if X.empty or y.empty:
        return None, None, None

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y)

    upcoming_drivers['Year'] = upcoming_year
    upcoming_drivers['Driver_encoded'] = le_driver.transform(upcoming_drivers['Abbreviation'].astype(str))
    upcoming_drivers['Team_encoded'] = le_team.transform(upcoming_drivers['TeamName'].astype(str))

    X_upcoming = upcoming_drivers[['Driver_encoded', 'Team_encoded', 'Year']]
    upcoming_drivers['Predicted Position'] = model_rf.predict(X_upcoming)

    upcoming_drivers.sort_values('Predicted Position', inplace=True)
    upcoming_drivers.reset_index(drop=True, inplace=True)

    upcoming_drivers['Time Gap (s)'] = [round(i * 2.5 + (i ** 1.1), 3) for i in range(len(upcoming_drivers))]
    upcoming_drivers['Predicted Finish Time'] = upcoming_drivers['Time Gap (s)'].apply(
        lambda t: f"+{t:.3f}s" if t > 0 else "Leader"
    )

    return upcoming_drivers, actual_results, show_actual

# --- Simulate Live Race Function ---
def simulate_live_race(drivers_df):
    """Simulate a live race with changing positions"""
    if 'race_lap' not in st.session_state:
        st.session_state.race_lap = 1
        st.session_state.race_data = drivers_df.copy()
        st.session_state.race_data['Current Gap'] = st.session_state.race_data['Time Gap (s)']

    # Update positions based on random events
    if st.session_state.race_lap > 1:
        # Randomly select drivers who might have position changes
        n_drivers = len(st.session_state.race_data)
        overtake_chance = np.random.random(n_drivers)

        for i in range(1, n_drivers):
            # 15% chance of position change
            if overtake_chance[i] < 0.15:
                # Only allowing to overtake the driver ahead
                if i > 0:
                    # Swap positions
                    temp_gap = st.session_state.race_data.iloc[i-1]['Current Gap']
                    st.session_state.race_data.iloc[i-1, st.session_state.race_data.columns.get_loc('Current Gap')] = \
                        st.session_state.race_data.iloc[i]['Current Gap']
                    st.session_state.race_data.iloc[i, st.session_state.race_data.columns.get_loc('Current Gap')] = temp_gap

    # Sort by current gap
    st.session_state.race_data = st.session_state.race_data.sort_values('Current Gap')

    # Update lap counter
    st.session_state.race_lap += 1

    return st.session_state.race_data, st.session_state.race_lap

# --- Team Colors Dictionary ---
team_colors = {
    'Red Bull': '#0600EF',
    'Mercedes': '#00D2BE',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Alpine': '#0090FF',
    'AlphaTauri': '#2B4562',
    'Aston Martin': '#006F62',
    'Williams': '#005AFF',
    'Alfa Romeo': '#900000',
    'Haas F1 Team': '#FFFFFF',
    'Racing Point': '#F596C8',
    'Renault': '#FFF500',
    'Toro Rosso': '#469BFF',
    'Force India': '#F596C8',
    'Sauber': '#9B0000',
    'Manor': '#323232',
    'Caterham': '#10518F',
    'Lotus': '#FFB800',
    'Marussia': '#6E0000',
    'HRT': '#333333',
    'Jaguar': '#006F62',
    'Jordan': '#EFC600',
    'BAR': '#B26500',
    'Arrows': '#FF8700',
    'Brawn GP': '#B5B5B5'
}

# --- Commentary Generation Function ---
def generate_race_commentary(lap_number, position_data, total_laps):
    """Generate AI commentary for the current race situation"""
    try:
        # Create a prompt for the AI based on the current race situation
        leader = position_data.iloc[0]['Abbreviation']
        second_place = position_data.iloc[1]['Abbreviation'] if len(position_data) > 1 else "N/A"
        gap = position_data.iloc[1]['Current Gap'] if len(position_data) > 1 else 0

        # Calculate race progress
        race_progress = (lap_number / total_laps) * 100

        # Find if there were any position changes
        position_changes = []
        if 'previous_positions' in st.session_state and lap_number > 1:
            current_positions = position_data['Abbreviation'].tolist()
            prev_positions = st.session_state.previous_positions

            for i, driver in enumerate(current_positions):
                if driver in prev_positions:
                    prev_idx = prev_positions.index(driver)
                    if prev_idx != i:
                        if prev_idx > i:
                            position_changes.append(f"{driver} gained {prev_idx - i} position(s)")
                        else:
                            position_changes.append(f"{driver} lost {i - prev_idx} position(s)")

        # Store current positions for next comparison
        st.session_state.previous_positions = position_data['Abbreviation'].tolist()

        # Generate commentary prompt
        commentary_prompt = f"""
        Generate a brief, exciting F1 commentary for lap {lap_number} of {total_laps} ({race_progress:.1f}% complete). Current race situation:
        - Current leader: {leader}
        - Second place: {second_place}
        - Gap between them: {gap:.2f} seconds

        {"- Position changes: " + ", ".join(position_changes) if position_changes else ""}

        Keep it short (2-3 sentences), exciting, and focused on the most interesting developments.
        If this is the final lap, make the commentary more dramatic and conclusive.
        """

        # Generate commentary using the model
        response = model.generate_content(commentary_prompt)
        return response.text
    except Exception as e:
        return f"Lap {lap_number}/{total_laps}: The race continues with {leader} in the lead."

# --- Main UI ---
st.markdown(
    """
    <div style="text-align: center; background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: #e10600; margin: 0; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.7);">🏁 F1Expert AI Dashboard</h1>
        <p style="color: #f2f2f2; font-size: 1.2em;">Your ultimate Formula 1 companion powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Layout: 3 Columns
col1, col2, col3 = st.columns([1, 2, 1])

# --- Column 1: Chatbot ---
with col1:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
            <h3 style="color: white; margin-top: 0;">💬 Ask RaceMaster</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    messages = fetch_conversation_history()
    for message in messages:
        if message['role'] == 'model':
            st.markdown(f'<div class="bot-message"><strong>RaceMaster:</strong> {message["parts"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'user' and "System prompt" not in message['parts']:
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["parts"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    # Chat input
if st.session_state.get("trivia_active", False):
    st.markdown(
        """
        <div style="background-color: rgba(225, 6, 0, 0.3); padding: 10px; border-radius: 5px; margin: 10px 0;">
            <p style="color: white; margin: 0;">Chat is disabled during Trivia Game. Complete or end the game to resume chatting.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Disabled input (for visual consistency)
    st.text_input("Chat disabled during trivia...", disabled=True, key="disabled_chat")
else:
    user_input = st.chat_input("Ask something about F1...")
    if user_input:
        messages = fetch_conversation_history()
        messages.append({"role": "user", "parts": user_input})

        with st.spinner("RaceMaster is thinking..."):
            response_text = response(messages)

        if isinstance(response_text, str):
            st.error(response_text)
        else:
            messages.append({"role": "model", "parts": response_text.text})
            st.rerun()

# --- Column 2: Prediction + Trivia Tabs ---
with col2:
    tab1, tab2 = st.tabs(["🏎️ Race Predictor", "❓ F1 Trivia"])

    with tab1:
        st.markdown(
            """
            <div style="background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
                <h3 style="color: white; margin-top: 0;">🔮 Predict Race Results</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container():
            col_year, col_gp = st.columns(2)

            with col_year:
                available_years = get_predictable_years()
                default_index = len(available_years) - 1 if available_years else 0
                selected_year = st.selectbox("Select Year", available_years, index=default_index)
            
            try:
                races, event_locations = get_predictable_gps(selected_year)
            
                if not races:
                    st.warning(f"No predictable races available for {selected_year}.")
                    selected_gp = None
                    selected_gp_location = None
                else:
                    with col_gp:
                        selected_gp = st.selectbox("Choose a Grand Prix", races)
            
                    selected_gp_location = event_locations.get(selected_gp)
            
                    if selected_gp:
                        st.markdown(f"""
                            <div style="background-color: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; margin: 10px 0;">
                                <p style="color: white; margin: 0;"><strong>📍 Circuit:</strong> {selected_gp_location or 'Unknown'}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error loading race schedule: {e}")
                selected_gp = None
                selected_gp_location = None

            if selected_gp and st.button("Run Prediction", key="predict_button"):
                with st.spinner("Running prediction model..."):
                    predictions, actual, has_real = predict_all_positions(selected_gp, selected_year, selected_gp_location)

                    if predictions is None or predictions.empty:
                        st.error(f"No usable race data found for {selected_gp} in {selected_year}. Try another year or GP.")
                    else:
                        st.success("✅ Prediction complete!")

                        # Store in session state for live simulation
                        st.session_state.predictions = predictions

                        # Show prediction table with styling
                        st.markdown("#### 🏁 Predicted Finish Order")
                        st.dataframe(
                            predictions[['Abbreviation', 'TeamName', 'Predicted Finish Time']],
                            hide_index=True,
                            use_container_width=True
                        )

                        # Visualization with team colors
                        st.markdown("#### 📊 Predicted Results Visualization")

                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_facecolor('#1e1e1e')
                        ax.set_facecolor('#1e1e1e')

                        # Set custom colors based on team
                        team_colors_list = [team_colors.get(team, '#777777') for team in predictions['TeamName']]

                        # Create horizontal bar chart
                        bars = ax.barh(
                            predictions['Abbreviation'],
                            predictions['Time Gap (s)'],
                            color=team_colors_list,
                            height=0.6
                        )

                        # Add team names as labels
                        for i, (bar, team) in enumerate(zip(bars, predictions['TeamName'])):
                            ax.text(
                                bar.get_width() + 0.5,
                                i,
                                team,
                                va='center',
                                color='white',
                                fontsize=8
                            )

                        # Styling
                        ax.set_title(f"Predicted Time Gaps: {selected_gp} {selected_year}", color='white', fontsize=14)
                        ax.set_xlabel('Gap to Leader (seconds)', color='white')
                        ax.set_ylabel('Drivers', color='white')
                        ax.tick_params(colors='white')
                        ax.grid(True, linestyle='--', alpha=0.3)

                        for spine in ax.spines.values():
                            spine.set_color('#333333')

                        plt.tight_layout()
                        st.pyplot(fig)

                        if has_real:
                            st.markdown("#### 🏁 Actual Results")
                            actual_sorted = actual.sort_values('Position')
                            st.dataframe(
                                actual_sorted[['Abbreviation', 'Position']],
                                hide_index=True,
                                use_container_width=True
                            )

                            # Comparison visualization
                            st.markdown("#### 📊 Prediction vs Actual Results")
                            comparison_df = predictions.merge(actual[['Abbreviation', 'Position']], on='Abbreviation', how='left')
                            comparison_df['Actual Position'] = comparison_df['Position'].fillna(0).astype(int)

                            fig, ax = plt.subplots(figsize=(10, 6))
                            fig.patch.set_facecolor('#1e1e1e')
                            ax.set_facecolor('#1e1e1e')

                            team_colors_scatter = [team_colors.get(team, '#777777') for team in comparison_df['TeamName']]

                            # Create scatter plot
                            ax.scatter(
                                comparison_df['Predicted Position'],
                                comparison_df['Actual Position'],
                                s=100,
                                c=team_colors_scatter,
                                alpha=0.7,
                                edgecolors='white'
                            )

                            # Add driver abbreviations as labels
                            for i, txt in enumerate(comparison_df['Abbreviation']):
                                ax.annotate(
                                    txt,
                                    (comparison_df['Predicted Position'].iloc[i], comparison_df['Actual Position'].iloc[i]),
                                    fontsize=9,
                                    color='white',
                                    ha='center',
                                    va='bottom',
                                    xytext=(0, 5),
                                    textcoords='offset points'
                                )

                            # Diagonal line (perfect prediction)
                            ax.plot([0, 20], [0, 20], 'r--', alpha=0.5)

                            # Styling
                            ax.set_title(f"Prediction vs Actual: {selected_gp} {selected_year}", color='white', fontsize=14)
                            ax.set_xlabel('Predicted Position', color='white')
                            ax.set_ylabel('Actual Position', color='white')
                            ax.tick_params(colors='white')
                            ax.grid(True, linestyle='--', alpha=0.3)

                            for spine in ax.spines.values():
                                spine.set_color('#333333')

                            plt.tight_layout()
                            st.pyplot(fig)

    with tab2:
        st.markdown(
            """
            <div style="background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
                <h3 style="color: white; margin-top: 0;">🎯 F1 Trivia Challenge</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Initialize trivia state
        if "trivia_questions" not in st.session_state:
            st.session_state.trivia_questions = []
            st.session_state.show_answers = {}
            st.session_state.user_score = 0
            st.session_state.total_questions = 0
            st.session_state.trivia_active = False

        col_trivia_btn, col_score = st.columns([2, 1])

        with col_trivia_btn:
            if st.button("Get New Trivia Questions"):
                with st.spinner("Generating challenging trivia questions..."):
                    st.session_state.trivia_questions = generate_f1_trivia()
                    st.session_state.show_answers = {i: False for i in range(len(st.session_state.trivia_questions))}
                    st.session_state.user_score = 0
                    st.session_state.total_questions = len(st.session_state.trivia_questions)
                    st.session_state.trivia_active = True

        with col_score:
            if st.session_state.total_questions > 0:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(0,50,0,0.7); padding: 10px; border-radius: 5px; text-align: center;">
                        <p style="color: white; margin: 0; font-size: 1.2em;">Score: {st.session_state.user_score}/{st.session_state.total_questions}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Display trivia questions
        if st.session_state.trivia_questions:
            for i, qa in enumerate(st.session_state.trivia_questions):
                with st.container():
                    st.markdown(
                        f"""
                        <div class="trivia-question">
                            <p style="color: white; font-size: 1.1em; font-weight: bold; margin-bottom: 5px;">Question {i+1}:</p>
                            <p style="color: white;">{qa['question']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if not st.session_state.show_answers[i]:
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            if f"user_answer_{i}" not in st.session_state:
                                st.session_state[f"user_answer_{i}"] = ""

                            user_answer = st.text_input("Your answer:", key=f"answer_input_{i}")

                        with col2:
                            if st.button("Submit", key=f"submit_{i}"):
                                st.session_state[f"user_answer_{i}"] = user_answer
                                st.session_state.show_answers[i] = True

                                # --- Fixed answer checking ---
                                correct_answer = qa['answer'].strip().lower()
                                user_input = user_answer.strip().lower()

                                if user_input == correct_answer:
                                    is_correct = True
                                else:
                                    key_terms = correct_answer.split()
                                    significant_terms = [term for term in key_terms if len(term) > 3]
                                    matches = [term for term in significant_terms if term in user_input]
                                    is_correct = bool(matches)

                                if is_correct:
                                    st.session_state.user_score += 1

                                st.rerun()

                    # Show answer after submission
                    if st.session_state.show_answers[i]:
                        user_response = st.session_state.get(f"user_answer_{i}", "")

                        correct_answer = qa['answer'].strip().lower()
                        user_input = user_response.strip().lower()

                        if user_input == correct_answer:
                            is_correct = True
                        else:
                            key_terms = correct_answer.split()
                            significant_terms = [term for term in key_terms if len(term) > 3]
                            matches = [term for term in significant_terms if term in user_input]
                            is_correct = bool(matches)

                        st.markdown(
                            f"""
                            <div class="trivia-answer" style="background-color: rgba({0 if is_correct else 225}, {100 if is_correct else 0}, 0, 0.5);">
                                <p style="color: white;"><strong>Your answer:</strong> {user_response}</p>
                                <p style="color: white;"><strong>Correct answer:</strong> {qa['answer']}</p>
                                <p style="color: white;"><strong>Result:</strong> {"✅ Correct!" if is_correct else "❌ Incorrect"}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.info("Click the button above to get some challenging F1 trivia questions!")

        if st.session_state.trivia_questions and st.session_state.trivia_active:
            if st.button("End Trivia Game"):
                st.session_state.trivia_active = False
                st.rerun()


# --- Column 3: Live Race Simulation ---
with col3:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
            <h3 style="color: white; margin-top: 0;">🌀 Live Race Tracking</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check if we have predictions to simulate
    if "predictions" in st.session_state:
        # Initialize race_active if not already set
        if "race_active" not in st.session_state:
            st.session_state.race_active = False
            st.session_state.commentary_history = []
            st.session_state.previous_positions = []
            st.session_state.race_points = {}
            st.session_state.auto_run = False

        # Race configuration
        if not st.session_state.race_active:
            col_laps, col_auto = st.columns(2)

            with col_laps:
                total_laps = st.slider("Number of Laps", min_value=5, max_value=50, value=10, step=5)
                st.session_state.total_laps = total_laps

            with col_auto:
                auto_run = st.checkbox("Auto Run Race", value=False)
                st.session_state.auto_run = auto_run

                if auto_run:
                    sim_speed = st.select_slider("Simulation Speed",
                                                options=["Slow", "Medium", "Fast"],
                                                value="Medium")
                    st.session_state.sim_speed = sim_speed

        # Add simulation controls
        col_start, col_next, col_score = st.columns([1, 1, 1])

        with col_start:
            if not st.session_state.race_active:
                if st.button("Start Race Simulation"):
                    st.session_state.race_active = True
                    st.session_state.race_lap = 0
                    st.session_state.race_data = st.session_state.predictions.copy()
                    st.session_state.race_data['Current Gap'] = st.session_state.race_data['Time Gap (s)']
                    st.session_state.commentary_history = []

                    # Initialize points dictionary for all drivers
                    st.session_state.race_points = {driver: 0 for driver in st.session_state.race_data['Abbreviation'].tolist()}
                    st.rerun()
            else:
                if st.button("Reset Simulation"):
                    st.session_state.race_active = False
                    st.session_state.commentary_history = []
                    st.session_state.auto_run = False
                    st.rerun()

        with col_next:
            if st.session_state.race_active and not st.session_state.auto_run:
                if st.button("Next Lap"):
                    # Simulate next lap if not reached total laps
                    if st.session_state.race_lap < st.session_state.total_laps:
                        updated_data, current_lap = simulate_live_race(st.session_state.race_data)

                        # Generate commentary for this lap
                        commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)

                        # Add commentary to history
                        st.session_state.commentary_history.append({
                            "lap": current_lap,
                            "text": commentary
                        })

                        # Update driver points for current positions
                        if current_lap == st.session_state.total_laps:
                            # Final lap - assign points
                            points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # F1 points system
                            for i, (idx, driver) in enumerate(updated_data.iterrows()):
                                if i < len(points_structure):
                                    st.session_state.race_points[driver['Abbreviation']] = points_structure[i]

                        st.rerun()

        with col_score:
            if st.session_state.race_active:
                if st.session_state.race_lap == st.session_state.total_laps:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(0, 100, 0, 0.7); padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
                            <p style="color: white; margin: 0; font-size: 1.2em;">Race Complete!</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(225, 6, 0, 0.7); padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
                            <p style="color: white; margin: 0; font-size: 1.2em;">Lap: {st.session_state.race_lap}/{st.session_state.total_laps}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Display race status if active
        if st.session_state.race_active:
            # Auto-advance simulation if auto_run is enabled
            if st.session_state.auto_run and st.session_state.race_lap < st.session_state.total_laps:
                # Determine delay based on simulation speed
                speed_delays = {"Slow": 3.0, "Medium": 1.5, "Fast": 0.5}
                delay = speed_delays.get(st.session_state.sim_speed, 1.5)

                # Add a progress indicator for the current lap
                progress_bar = st.progress(0)
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(delay/100)

                # Simulate next lap
                updated_data, current_lap = simulate_live_race(st.session_state.race_data)

                # Generate commentary for this lap
                commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)

                # Add commentary to history
                st.session_state.commentary_history.append({
                    "lap": current_lap,
                    "text": commentary
                })

                # Update driver points for current positions
                if current_lap == st.session_state.total_laps:
                    # Final lap - assign points
                    points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # F1 points system
                    for i, (idx, driver) in enumerate(updated_data.iterrows()):
                        if i < len(points_structure):
                            st.session_state.race_points[driver['Abbreviation']] = points_structure[i]

                st.rerun()

            # Display two sections side by side
            col_standings, col_points = st.columns(2)

            with col_standings:
                # Display live standings
                st.markdown("#### 🏁 Current Race Standings")

                for i, (idx, driver) in enumerate(st.session_state.race_data.iterrows()):
                    position = i + 1
                    time_gap = driver['Current Gap']
                    time_display = "Leader" if i == 0 else f"+{time_gap:.3f}s"

                    # Get team color
                    team_color = team_colors.get(driver['TeamName'], '#777777')

                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left: 4px solid {team_color}">
                            <span class="position">{position}</span>
                            <span class="driver-name">{driver['Abbreviation']}</span>
                            <span class="driver-time">{time_display}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_points:
                # Display points table
                st.markdown("#### 🏆 Championship Points")

                # Sort points by value (highest first)
                sorted_points = {k: v for k, v in sorted(st.session_state.race_points.items(),
                                                        key=lambda item: item[1], reverse=True)}

                for driver, points in sorted_points.items():
                    # Find the team color for this driver
                    driver_team = st.session_state.race_data[st.session_state.race_data['Abbreviation'] == driver]['TeamName'].values
                    team_color = team_colors.get(driver_team[0] if len(driver_team) > 0 else '', '#777777')

                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left: 4px solid {team_color}">
                            <span class="driver-name">{driver}</span>
                            <span class="position">{points} pts</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Display commentary section
            st.markdown("#### 🎙️ Race Commentary")

            commentary_container = st.container()
            with commentary_container:
                # Create a styled container for commentary
                st.markdown(
                    """
                    <div style="background-color: rgba(0,0,0,0.8); border-radius: 10px; padding: 10px; max-height: 300px; overflow-y: auto;">
                    """,
                    unsafe_allow_html=True
                )

                if st.session_state.commentary_history:
                    for comment in reversed(st.session_state.commentary_history):  # Show most recent first
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(30,30,30,0.7); padding: 8px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #e10600;">
                                <p style="color: #e10600; margin: 0; font-weight: bold;">Lap {comment['lap']}/{st.session_state.total_laps}</p>
                                <p style="color: white; margin: 0;">{comment['text']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        """
                        <p style="color: #aaaaaa; text-align: center;">Commentary will appear once the race begins.</p>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Predict a race first, then start the simulation to see the live race tracking!")

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
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    h1, h2, h3, h4 {
        color: #f2f2f2;
        font-family: 'Helvetica', sans-serif;
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
    .driver-card {
        background: linear-gradient(145deg, rgba(30,30,30,0.8), rgba(60,60,60,0.8));
        border-radius: 8px;
        padding: 8px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .position {
        font-weight: bold;
        font-size: 1.2rem;
        color: #e10600;
    }
    .driver-name {
        font-weight: bold;
        color: white;
    }
    .driver-time {
        color: #cccccc;
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
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
def get_ai_response(messages):
    try:
        resp = model.generate_content(messages)
        return resp
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
    trivia_prompt = """
    Generate exactly 5 unique Formula 1 trivia questions and very short, direct answers.
    Answer MUST be very short (driver name, team, year, number, etc.) without explanation.
    Format the output as JSON inside ```json ``` block, like:

    ```json
    [
        {"question": "Who won the 2021 F1 World Championship?", "answer": "Max Verstappen"},
        {"question": "Which team has the most Constructors Championships?", "answer": "Ferrari"}
    ]
    ```
    Only include short, direct answers. No explanations, no extra sentences.
    """
    try:
        resp = model.generate_content(trivia_prompt)
        import json, re
        text = resp.text
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text
        json_str = json_str.replace('\\n', '').replace('\\', '')
        questions = json.loads(json_str)
        for qa in questions:
            if 'answer' in qa:
                qa['answer'] = qa['answer'].split('.')[0].strip()
        return questions
    except Exception as e:
        st.error(f"Trivia generation error: {str(e)}")
        return [
            {"question": "Who holds the most F1 World Championships?", "answer": "Lewis Hamilton"},
            {"question": "Which team is nicknamed the Silver Arrows?", "answer": "Mercedes"},
            {"question": "Which country hosts the Monaco Grand Prix?", "answer": "Monaco"},
            {"question": "Who was Ferraris first F1 world champion?", "answer": "Alberto Ascari"},
            {"question": "What tire supplier is used in Formula 1?", "answer": "Pirelli"}
        ]

# --- Get Latest Year ---
def get_latest_year():
    return datetime.now().year

# --- Team Colors Dictionary ---
team_colors = {
    'Red Bull Racing': '#0600EF',
    'Red Bull': '#0600EF',
    'Mercedes': '#00D2BE',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Alpine': '#0090FF',
    'AlphaTauri': '#2B4562',
    'RB F1 Team': '#2B4562',
    'Aston Martin': '#006F62',
    'Williams': '#005AFF',
    'Alfa Romeo': '#900000',
    'Haas F1 Team': '#FFFFFF',
    'Haas': '#FFFFFF',
    'Racing Point': '#F596C8',
    'Renault': '#FFF500',
    'Toro Rosso': '#469BFF',
    'Force India': '#F596C8',
    'Sauber': '#9B0000',
    'Kick Sauber': '#00CF46',
    'Manor': '#323232',
    'Caterham': '#10518F',
    'Lotus': '#FFB800',
    'Marussia': '#6E0000',
    'HRT': '#333333',
}

# ─────────────────────────────────────────────
# SMART DATA FETCHING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def discover_available_years():
    """
    Scan from 2018 to current year and return a list of years
    for which FastF1 can successfully fetch the event schedule.
    Result is cached so it only runs once per session.
    """
    available = []
    current = get_latest_year()
    for yr in range(2018, current + 1):
        try:
            schedule = fastf1.get_event_schedule(yr)
            if schedule is not None and len(schedule) > 0:
                available.append(yr)
        except Exception:
            continue
    return available


@st.cache_data(show_spinner=False)
def get_schedule_for_year(year):
    """Return the race schedule for a given year, cached."""
    try:
        schedule = fastf1.get_event_schedule(year)
        races = schedule[
            (schedule['EventFormat'].notna()) &
            (schedule['EventFormat'] != 'testing') &
            (schedule['RoundNumber'] > 0) &
            (~schedule['EventName'].str.contains(
                "Testing|Test|Pre-Season|Track Session|Filming|Shakedown",
                case=False, na=False
            ))
        ]
        return races
    except Exception:
        return None


def load_race_session(year, gp_name):
    """
    Try multiple loading strategies to get race results.
    Returns (results_df, error_string).
    """
    # Strategy 1: load with minimal flags (fastest)
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results[['Abbreviation', 'Position', 'TeamName']].copy()
        results['Year'] = year
        results = results.dropna(subset=['Abbreviation', 'Position', 'TeamName'])
        results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
        results = results.dropna(subset=['Position'])
        if len(results) > 0:
            return results, None
    except Exception as e1:
        pass

    # Strategy 2: load with no flags at all (some versions need this)
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load()
        results = session.results[['Abbreviation', 'Position', 'TeamName']].copy()
        results['Year'] = year
        results = results.dropna(subset=['Abbreviation', 'Position', 'TeamName'])
        results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
        results = results.dropna(subset=['Position'])
        if len(results) > 0:
            return results, None
    except Exception as e2:
        pass

    # Strategy 3: try by round number instead of name
    try:
        schedule = fastf1.get_event_schedule(year)
        matches = schedule[schedule['EventName'].str.contains(
            gp_name.replace(' Grand Prix', '').strip(), case=False, na=False
        )]
        if len(matches) > 0:
            round_num = int(matches.iloc[0]['RoundNumber'])
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results[['Abbreviation', 'Position', 'TeamName']].copy()
            results['Year'] = year
            results = results.dropna(subset=['Abbreviation', 'Position', 'TeamName'])
            results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
            results = results.dropna(subset=['Position'])
            if len(results) > 0:
                return results, None
    except Exception as e3:
        return None, str(e3)

    return None, "All loading strategies failed"


def build_driver_pool_from_schedule(year, gp_name):
    """
    When we can't load session results, build a plausible driver list
    from the ergast/schedule API which is lighter weight.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        # Return known F1 2024/2025 driver pool as fallback
        default_drivers = [
            {"Abbreviation": "VER", "TeamName": "Red Bull Racing"},
            {"Abbreviation": "PER", "TeamName": "Red Bull Racing"},
            {"Abbreviation": "HAM", "TeamName": "Mercedes"},
            {"Abbreviation": "RUS", "TeamName": "Mercedes"},
            {"Abbreviation": "LEC", "TeamName": "Ferrari"},
            {"Abbreviation": "SAI", "TeamName": "Ferrari"},
            {"Abbreviation": "NOR", "TeamName": "McLaren"},
            {"Abbreviation": "PIA", "TeamName": "McLaren"},
            {"Abbreviation": "ALO", "TeamName": "Aston Martin"},
            {"Abbreviation": "STR", "TeamName": "Aston Martin"},
            {"Abbreviation": "GAS", "TeamName": "Alpine"},
            {"Abbreviation": "OCO", "TeamName": "Alpine"},
            {"Abbreviation": "TSU", "TeamName": "RB F1 Team"},
            {"Abbreviation": "RIC", "TeamName": "RB F1 Team"},
            {"Abbreviation": "ALB", "TeamName": "Williams"},
            {"Abbreviation": "SAR", "TeamName": "Williams"},
            {"Abbreviation": "BOT", "TeamName": "Kick Sauber"},
            {"Abbreviation": "ZHO", "TeamName": "Kick Sauber"},
            {"Abbreviation": "MAG", "TeamName": "Haas F1 Team"},
            {"Abbreviation": "HUL", "TeamName": "Haas F1 Team"},
        ]
        return pd.DataFrame(default_drivers)
    except Exception:
        return None


def predict_all_positions(gp_name, upcoming_year, available_years):
    """
    Build a Random Forest model from all available historical data
    for gp_name, then predict the finish order for upcoming_year.
    """

    # ── 1. Collect historical race data ──────────────────────────────
    all_races = []
    years_loaded = []
    history_years = sorted([y for y in available_years if y != upcoming_year], reverse=True)

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    total = len(history_years)

    for idx, year in enumerate(history_years):
        pct = int((idx / max(total, 1)) * 100)
        progress_bar.progress(pct)
        status_placeholder.info(f"⏳ Loading historical race data… ({idx + 1}/{total})")

        results, err = load_race_session(year, gp_name)
        if results is not None and len(results) > 0:
            all_races.append(results)
            years_loaded.append(year)
            print(f"✅ Loaded {gp_name} {year} — {len(results)} drivers")
        else:
            print(f"❌ Skipped {gp_name} {year}: {err}")

    progress_bar.progress(100)
    status_placeholder.empty()
    progress_bar.empty()

    # ── 2. Handle no historical data at all ──────────────────────────
    if not all_races:
        st.warning(
            f"⚠️ Could not load race session data for **{gp_name}**. "
            f"FastF1 may be rate-limiting or the session files are unavailable on Streamlit Cloud. "
            f"Try a different Grand Prix or wait a moment and retry."
        )
        return None, None, None

    df = pd.concat(all_races, ignore_index=True)
    df.dropna(inplace=True)

    if df.empty:
        st.warning("⚠️ Historical data is empty after cleaning. Try a different Grand Prix.")
        return None, None, None

    latest_hist_year = max(years_loaded)
    print(f"✅ Historical data ready — years: {years_loaded}, rows: {len(df)}")

    # ── 3. Try to load the target year's actual results ───────────────
    actual_results = None
    show_actual    = False
    upcoming_drivers = None

    status_placeholder3 = st.empty()
    status_placeholder3.info("⏳ Loading target race data…")

    target_results, err = load_race_session(upcoming_year, gp_name)
    status_placeholder3.empty()

    if target_results is not None and len(target_results) > 0:
        actual_results   = target_results[['Abbreviation', 'Position']]
        upcoming_drivers = target_results[['Abbreviation', 'TeamName']].drop_duplicates()
        show_actual      = True
        print(f"✅ Loaded actual results for {upcoming_year}")
    else:
        print(f"❌ Could not load {upcoming_year} session: {err}")
        # Use latest historical year's drivers as the lineup
        upcoming_drivers = (
            df[df['Year'] == latest_hist_year][['Abbreviation', 'TeamName']]
            .drop_duplicates()
        )

        # Last resort — use built-in driver pool
        if upcoming_drivers is None or upcoming_drivers.empty:
            upcoming_drivers = build_driver_pool_from_schedule(upcoming_year, gp_name)

    if upcoming_drivers is None or upcoming_drivers.empty:
        st.warning("⚠️ Could not determine the driver lineup. Try a different year or Grand Prix.")
        return None, None, None

    # ── 4. Encode & train ─────────────────────────────────────────────
    all_drivers = list(set(df['Abbreviation'].tolist() + upcoming_drivers['Abbreviation'].tolist()))
    all_teams   = list(set(df['TeamName'].tolist()    + upcoming_drivers['TeamName'].tolist()))

    le_driver = LabelEncoder().fit(all_drivers)
    le_team   = LabelEncoder().fit(all_teams)

    df['Driver_encoded'] = le_driver.transform(df['Abbreviation'])
    df['Team_encoded']   = le_team.transform(df['TeamName'])

    X = df[['Driver_encoded', 'Team_encoded', 'Year']]
    y = df['Position']

    if len(X) < 3:
        st.warning("⚠️ Not enough data rows to train a model. Try a Grand Prix with more history.")
        return None, None, None

    model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    model_rf.fit(X, y)

    # ── 5. Predict ────────────────────────────────────────────────────
    upcoming_drivers = upcoming_drivers.copy()
    upcoming_drivers['Year'] = upcoming_year

    def safe_transform(le, values):
        known = set(le.classes_)
        return [le.transform([v])[0] if v in known else 0 for v in values]

    upcoming_drivers['Driver_encoded'] = safe_transform(le_driver, upcoming_drivers['Abbreviation'])
    upcoming_drivers['Team_encoded']   = safe_transform(le_team,   upcoming_drivers['TeamName'])

    X_upcoming = upcoming_drivers[['Driver_encoded', 'Team_encoded', 'Year']]
    upcoming_drivers['Predicted Position'] = model_rf.predict(X_upcoming)

    upcoming_drivers.sort_values('Predicted Position', inplace=True)
    upcoming_drivers.reset_index(drop=True, inplace=True)

    # ── 6. Simulate time gaps ─────────────────────────────────────────
    upcoming_drivers['Time Gap (s)'] = [
        round(i * 2.5 + (i ** 1.1), 3) for i in range(len(upcoming_drivers))
    ]
    upcoming_drivers['Predicted Finish Time'] = upcoming_drivers['Time Gap (s)'].apply(
        lambda t: "Leader" if t == 0 else f"+{t:.3f}s"
    )

    print(f"✅ Prediction complete — {len(upcoming_drivers)} drivers")
    return upcoming_drivers, actual_results, show_actual


# --- Simulate Live Race Function ---
def simulate_live_race(drivers_df):
    if 'race_lap' not in st.session_state:
        st.session_state.race_lap = 1
        st.session_state.race_data = drivers_df.copy()
        st.session_state.race_data['Current Gap'] = st.session_state.race_data['Time Gap (s)']

    if st.session_state.race_lap > 1:
        n_drivers = len(st.session_state.race_data)
        overtake_chance = np.random.random(n_drivers)
        for i in range(1, n_drivers):
            if overtake_chance[i] < 0.15 and i > 0:
                temp_gap = st.session_state.race_data.iloc[i-1]['Current Gap']
                st.session_state.race_data.iloc[i-1, st.session_state.race_data.columns.get_loc('Current Gap')] = \
                    st.session_state.race_data.iloc[i]['Current Gap']
                st.session_state.race_data.iloc[i, st.session_state.race_data.columns.get_loc('Current Gap')] = temp_gap

    st.session_state.race_data = st.session_state.race_data.sort_values('Current Gap')
    st.session_state.race_lap += 1
    return st.session_state.race_data, st.session_state.race_lap


# --- Commentary Generation ---
def generate_race_commentary(lap_number, position_data, total_laps):
    try:
        leader = position_data.iloc[0]['Abbreviation']
        second_place = position_data.iloc[1]['Abbreviation'] if len(position_data) > 1 else "N/A"
        gap = position_data.iloc[1]['Current Gap'] if len(position_data) > 1 else 0
        race_progress = (lap_number / total_laps) * 100

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

        st.session_state.previous_positions = position_data['Abbreviation'].tolist()

        commentary_prompt = f"""
        Generate a brief, exciting F1 commentary for lap {lap_number} of {total_laps} ({race_progress:.1f}% complete). Current race situation:
        - Current leader: {leader}
        - Second place: {second_place}
        - Gap between them: {gap:.2f} seconds
        {"- Position changes: " + ", ".join(position_changes) if position_changes else ""}
        Keep it short (2-3 sentences), exciting, and focused on the most interesting developments.
        If this is the final lap, make the commentary more dramatic and conclusive.
        """
        resp = model.generate_content(commentary_prompt)
        return resp.text
    except Exception:
        return f"Lap {lap_number}/{total_laps}: Racing continues!"


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; background-color:rgba(0,0,0,0.7); padding:20px; border-radius:15px; margin-bottom:20px;">
        <h1 style="color:#e10600; margin:0; font-size:3em; text-shadow:2px 2px 4px rgba(0,0,0,0.7);">🏁 F1Expert AI Dashboard</h1>
        <p style="color:#f2f2f2; font-size:1.2em;">Your ultimate Formula 1 companion powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Discover available years once ────────────────────────────────────
with st.spinner("🔍 Detecting available F1 data years…"):
    AVAILABLE_YEARS = discover_available_years()

if not AVAILABLE_YEARS:
    st.error("❌ Could not connect to FastF1 data source. Please check your internet connection or try again later.")
    st.stop()

MIN_YEAR = min(AVAILABLE_YEARS)
MAX_YEAR = max(AVAILABLE_YEARS)

# Layout: 3 Columns
col1, col2, col3 = st.columns([1, 2, 1])

# ── Column 1: Chatbot ─────────────────────────────────────────────────
with col1:
    st.markdown(
        """
        <div style="background-color:rgba(0,0,0,0.7); padding:15px; border-radius:10px; border-top:4px solid #e10600;">
            <h3 style="color:white; margin-top:0;">💬 Ask RaceMaster</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    messages = fetch_conversation_history()
    for message in messages:
        if message['role'] == 'model':
            st.markdown(f'<div class="bot-message"><strong>RaceMaster:</strong> {message["parts"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'user' and "System prompt" not in message['parts']:
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["parts"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("trivia_active", False):
    st.markdown(
        """
        <div style="background-color:rgba(225,6,0,0.3); padding:10px; border-radius:5px; margin:10px 0;">
            <p style="color:white; margin:0;">Chat is disabled during Trivia Game.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.text_input("Chat disabled during trivia...", disabled=True, key="disabled_chat")
else:
    user_input = st.chat_input("Ask something about F1...")
    if user_input:
        messages = fetch_conversation_history()
        messages.append({"role": "user", "parts": user_input})
        with st.spinner("RaceMaster is thinking..."):
            response_text = get_ai_response(messages)
        if isinstance(response_text, str):
            st.error(response_text)
        else:
            messages.append({"role": "model", "parts": response_text.text})
            st.rerun()

# ── Column 2: Prediction + Trivia ────────────────────────────────────
with col2:
    tab1, tab2 = st.tabs(["🏎️ Race Predictor", "❓ F1 Trivia"])

    # ── Tab 1: Race Predictor ─────────────────────────────────────────
    with tab1:
        st.markdown(
            """
            <div style="background-color:rgba(0,0,0,0.7); padding:15px; border-radius:10px; border-top:4px solid #e10600;">
                <h3 style="color:white; margin-top:0;">🔮 Predict Race Results</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show user what data is available
        st.markdown(
            f"<p style='color:#aaaaaa; font-size:0.85em;'>📡 Data available for years: <strong style='color:#e10600'>{', '.join(map(str, AVAILABLE_YEARS))}</strong></p>",
            unsafe_allow_html=True
        )

        with st.container():
            col_year, col_gp = st.columns(2)

            with col_year:
                selected_year = st.slider(
                    "Select Year",
                    min_value=MIN_YEAR,
                    max_value=MAX_YEAR,
                    value=MAX_YEAR
                )

            # Load schedule for selected year
            schedule = get_schedule_for_year(selected_year)

            if schedule is not None and len(schedule) > 0:
                races = schedule['EventName'].tolist()
                event_locations = dict(zip(schedule['EventName'], schedule['Location']))

                with col_gp:
                    selected_gp = st.selectbox("Choose a Grand Prix", races)

                if selected_gp:
                    st.markdown(
                        f"""
                        <div style="background-color:rgba(0,0,0,0.5); padding:10px; border-radius:5px; margin:10px 0;">
                            <p style="color:white; margin:0;"><strong>📍 Circuit:</strong> {event_locations.get(selected_gp, 'Unknown')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning(f"⚠️ Could not load race schedule for {selected_year}.")
                selected_gp = None

            if selected_gp and st.button("Run Prediction", key="predict_button"):
                with st.spinner("Running prediction model…"):
                    predictions, actual, has_real = predict_all_positions(
                        selected_gp, selected_year, AVAILABLE_YEARS
                    )

                if predictions is not None:
                    st.success("✅ Prediction complete!")
                    st.session_state.predictions = predictions

                    # Predicted finish order
                    st.markdown("#### 🏁 Predicted Finish Order")
                    st.dataframe(
                        predictions[['Abbreviation', 'TeamName', 'Predicted Finish Time']],
                        hide_index=True,
                        use_container_width=True
                    )

                    # Bar chart
                    st.markdown("#### 📊 Predicted Time Gaps")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#1e1e1e')
                    ax.set_facecolor('#1e1e1e')

                    team_colors_list = [team_colors.get(team, '#777777') for team in predictions['TeamName']]
                    bars = ax.barh(predictions['Abbreviation'], predictions['Time Gap (s)'],
                                   color=team_colors_list, height=0.6)

                    for i, (bar, team) in enumerate(zip(bars, predictions['TeamName'])):
                        ax.text(bar.get_width() + 0.5, i, team, va='center', color='white', fontsize=8)

                    ax.set_title(f"Predicted Time Gaps: {selected_gp} {selected_year}", color='white', fontsize=14)
                    ax.set_xlabel('Gap to Leader (seconds)', color='white')
                    ax.set_ylabel('Drivers', color='white')
                    ax.tick_params(colors='white')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    for spine in ax.spines.values():
                        spine.set_color('#333333')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Actual results comparison
                    if has_real and actual is not None:
                        st.markdown("#### 🏁 Actual Results")
                        actual_sorted = actual.sort_values('Position')
                        st.dataframe(actual_sorted[['Abbreviation', 'Position']],
                                     hide_index=True, use_container_width=True)

                        st.markdown("#### 📊 Prediction vs Actual")
                        comparison_df = predictions.merge(
                            actual[['Abbreviation', 'Position']], on='Abbreviation', how='left'
                        )
                        comparison_df['Actual Position'] = comparison_df['Position'].fillna(0).astype(int)

                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        fig2.patch.set_facecolor('#1e1e1e')
                        ax2.set_facecolor('#1e1e1e')

                        scatter_colors = [team_colors.get(t, '#777777') for t in comparison_df['TeamName']]
                        ax2.scatter(comparison_df['Predicted Position'], comparison_df['Actual Position'],
                                    s=100, c=scatter_colors, alpha=0.7, edgecolors='white')

                        for i, txt in enumerate(comparison_df['Abbreviation']):
                            ax2.annotate(txt,
                                (comparison_df['Predicted Position'].iloc[i],
                                 comparison_df['Actual Position'].iloc[i]),
                                fontsize=9, color='white', ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points')

                        ax2.plot([0, 22], [0, 22], 'r--', alpha=0.5)
                        ax2.set_title(f"Prediction vs Actual: {selected_gp} {selected_year}", color='white', fontsize=14)
                        ax2.set_xlabel('Predicted Position', color='white')
                        ax2.set_ylabel('Actual Position', color='white')
                        ax2.tick_params(colors='white')
                        ax2.grid(True, linestyle='--', alpha=0.3)
                        for spine in ax2.spines.values():
                            spine.set_color('#333333')
                        plt.tight_layout()
                        st.pyplot(fig2)
                else:
                    st.error("❌ Unable to generate predictions. Please try a different Grand Prix or year.")

    # ── Tab 2: F1 Trivia ──────────────────────────────────────────────
    with tab2:
        st.markdown(
            """
            <div style="background-color:rgba(0,0,0,0.7); padding:15px; border-radius:10px; border-top:4px solid #e10600;">
                <h3 style="color:white; margin-top:0;">🎯 F1 Trivia Challenge</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if "trivia_questions" not in st.session_state:
            st.session_state.trivia_questions = []
            st.session_state.show_answers = {}
            st.session_state.user_score = 0
            st.session_state.total_questions = 0
            st.session_state.trivia_active = False

        col_trivia_btn, col_score = st.columns([2, 1])

        with col_trivia_btn:
            if st.button("Get New Trivia Questions"):
                with st.spinner("Generating trivia questions..."):
                    st.session_state.trivia_questions = generate_f1_trivia()
                    st.session_state.show_answers = {i: False for i in range(len(st.session_state.trivia_questions))}
                    st.session_state.user_score = 0
                    st.session_state.total_questions = len(st.session_state.trivia_questions)
                    st.session_state.trivia_active = True

        with col_score:
            if st.session_state.total_questions > 0:
                st.markdown(
                    f"""
                    <div style="background-color:rgba(0,50,0,0.7); padding:10px; border-radius:5px; text-align:center;">
                        <p style="color:white; margin:0; font-size:1.2em;">Score: {st.session_state.user_score}/{st.session_state.total_questions}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if st.session_state.trivia_questions:
            for i, qa in enumerate(st.session_state.trivia_questions):
                st.markdown(
                    f"""
                    <div class="trivia-question">
                        <p style="color:white; font-size:1.1em; font-weight:bold; margin-bottom:5px;">Question {i+1}:</p>
                        <p style="color:white;">{qa['question']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if not st.session_state.show_answers.get(i, False):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        user_answer = st.text_input("Your answer:", key=f"answer_input_{i}")
                    with c2:
                        if st.button("Submit", key=f"submit_{i}"):
                            st.session_state[f"user_answer_{i}"] = user_answer
                            st.session_state.show_answers[i] = True
                            correct = qa['answer'].strip().lower()
                            given  = user_answer.strip().lower()
                            if given == correct:
                                is_correct = True
                            else:
                                key_terms = [t for t in correct.split() if len(t) > 3]
                                is_correct = bool([t for t in key_terms if t in given])
                            if is_correct:
                                st.session_state.user_score += 1
                            st.rerun()

                if st.session_state.show_answers.get(i, False):
                    user_response = st.session_state.get(f"user_answer_{i}", "")
                    correct = qa['answer'].strip().lower()
                    given  = user_response.strip().lower()
                    if given == correct:
                        is_correct = True
                    else:
                        key_terms = [t for t in correct.split() if len(t) > 3]
                        is_correct = bool([t for t in key_terms if t in given])

                    bg = "rgba(0,100,0,0.5)" if is_correct else "rgba(225,0,0,0.5)"
                    st.markdown(
                        f"""
                        <div style="background-color:{bg}; padding:10px; border-radius:5px; margin-top:5px;">
                            <p style="color:white;"><strong>Your answer:</strong> {user_response}</p>
                            <p style="color:white;"><strong>Correct answer:</strong> {qa['answer']}</p>
                            <p style="color:white;"><strong>Result:</strong> {"✅ Correct!" if is_correct else "❌ Incorrect"}</p>
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


# ── Column 3: Live Race Simulation ────────────────────────────────────
with col3:
    st.markdown(
        """
        <div style="background-color:rgba(0,0,0,0.7); padding:15px; border-radius:10px; border-top:4px solid #e10600;">
            <h3 style="color:white; margin-top:0;">🌀 Live Race Tracking</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "predictions" in st.session_state:
        if "race_active" not in st.session_state:
            st.session_state.race_active = False
            st.session_state.commentary_history = []
            st.session_state.previous_positions = []
            st.session_state.race_points = {}
            st.session_state.auto_run = False

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

        col_start, col_next, col_score = st.columns([1, 1, 1])

        with col_start:
            if not st.session_state.race_active:
                if st.button("Start Race Simulation"):
                    st.session_state.race_active = True
                    st.session_state.race_lap = 0
                    st.session_state.race_data = st.session_state.predictions.copy()
                    st.session_state.race_data['Current Gap'] = st.session_state.race_data['Time Gap (s)']
                    st.session_state.commentary_history = []
                    st.session_state.race_points = {
                        d: 0 for d in st.session_state.race_data['Abbreviation'].tolist()
                    }
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
                    if st.session_state.race_lap < st.session_state.total_laps:
                        updated_data, current_lap = simulate_live_race(st.session_state.race_data)
                        commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)
                        st.session_state.commentary_history.append({"lap": current_lap, "text": commentary})
                        if current_lap == st.session_state.total_laps:
                            points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                            for i, (idx, driver) in enumerate(updated_data.iterrows()):
                                if i < len(points_structure):
                                    st.session_state.race_points[driver['Abbreviation']] = points_structure[i]
                        st.rerun()

        with col_score:
            if st.session_state.race_active:
                if st.session_state.race_lap == st.session_state.total_laps:
                    st.markdown(
                        """
                        <div style="background-color:rgba(0,100,0,0.7); padding:10px; border-radius:5px; text-align:center; margin:10px 0;">
                            <p style="color:white; margin:0; font-size:1.2em;">🏆 Race Complete!</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color:rgba(225,6,0,0.7); padding:10px; border-radius:5px; text-align:center; margin:10px 0;">
                            <p style="color:white; margin:0; font-size:1.2em;">Lap: {st.session_state.race_lap}/{st.session_state.total_laps}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if st.session_state.race_active:
            if st.session_state.auto_run and st.session_state.race_lap < st.session_state.total_laps:
                speed_delays = {"Slow": 3.0, "Medium": 1.5, "Fast": 0.5}
                delay = speed_delays.get(st.session_state.get('sim_speed', 'Medium'), 1.5)
                progress_bar = st.progress(0)
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(delay / 100)
                updated_data, current_lap = simulate_live_race(st.session_state.race_data)
                commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)
                st.session_state.commentary_history.append({"lap": current_lap, "text": commentary})
                if current_lap == st.session_state.total_laps:
                    points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                    for i, (idx, driver) in enumerate(updated_data.iterrows()):
                        if i < len(points_structure):
                            st.session_state.race_points[driver['Abbreviation']] = points_structure[i]
                st.rerun()

            col_standings, col_points = st.columns(2)

            with col_standings:
                st.markdown("#### 🏁 Current Standings")
                for i, (idx, driver) in enumerate(st.session_state.race_data.iterrows()):
                    position = i + 1
                    time_gap = driver['Current Gap']
                    time_display = "Leader" if i == 0 else f"+{time_gap:.3f}s"
                    team_color = team_colors.get(driver['TeamName'], '#777777')
                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left:4px solid {team_color}">
                            <span class="position">{position}</span>
                            <span class="driver-name">{driver['Abbreviation']}</span>
                            <span class="driver-time">{time_display}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_points:
                st.markdown("#### 🏆 Points")
                sorted_points = dict(sorted(st.session_state.race_points.items(),
                                            key=lambda item: item[1], reverse=True))
                for driver, points in sorted_points.items():
                    driver_team = st.session_state.race_data[
                        st.session_state.race_data['Abbreviation'] == driver
                    ]['TeamName'].values
                    team_color = team_colors.get(driver_team[0] if len(driver_team) > 0 else '', '#777777')
                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left:4px solid {team_color}">
                            <span class="driver-name">{driver}</span>
                            <span class="position">{points} pts</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("#### 🎙️ Commentary")
            st.markdown('<div style="background-color:rgba(0,0,0,0.8); border-radius:10px; padding:10px; max-height:300px; overflow-y:auto;">', unsafe_allow_html=True)
            if st.session_state.commentary_history:
                for comment in reversed(st.session_state.commentary_history):
                    st.markdown(
                        f"""
                        <div style="background-color:rgba(30,30,30,0.7); padding:8px; border-radius:8px; margin-bottom:8px; border-left:3px solid #e10600;">
                            <p style="color:#e10600; margin:0; font-weight:bold;">Lap {comment['lap']}/{st.session_state.total_laps}</p>
                            <p style="color:white; margin:0;">{comment['text']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<p style="color:#aaaaaa; text-align:center;">Commentary will appear once the race begins.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Predict a race first, then start the simulation!")


# Import Streamlit to build the web application
import streamlit as st

# Import joblib to load the trained machine learning model
import joblib

# Configure the browser tab and page layout
st.set_page_config(
    page_title="BikeFlow Seoul",
    page_icon="🚲",
    layout="wide"
)

# Apply custom CSS to style the application
st.markdown(
    """
    <style>
    /* Set the main page background to dark navy */
    .stApp {
        background-color: #102B40;
        color: #FFFFFF;
    }

    /* Style the main heading */
    .stApp h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem !important;
        font-weight: 650;
    }

    /* Style paragraph text */
    p {
        color: #D0E2E9;
    }

    /* Add space around the main content */
    .block-container {
        padding-top: 4rem;
        padding-bottom: 3rem;
    }
    /* Improve the contrast of the primary prediction button */
    button[data-testid="stBaseButton-primary"] p {
        color: #102B40 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the application name
st.title("BikeFlow Seoul")


# Import Image from Pillow to crop the photograph
from PIL import Image

# Open the photograph stored in the assets folder
seoul_image = Image.open("assets/seoul_night.jpg")

# Get the original image dimensions
image_width, image_height = seoul_image.size

# Define a wide panoramic crop (approximately 5:1)
crop_height = min(image_height, image_width // 3)

# Position the crop near the lower part of the image
top = int((image_height - crop_height) * 0.45)

# Crop the image while keeping its full width
seoul_banner = seoul_image.crop(
    (0, top, image_width, top + crop_height)
)

# Display the cropped panorama
st.image(seoul_banner, width="stretch")

# Credit the photographer
st.caption("Photography: Clark Gu / Unsplash")

# Introduce the purpose of the dashboard
st.subheader("Discover the rhythm of Seoul.")

# Explain what visitors can explore
st.write(
    "Explore bike rental patterns and predict hourly demand "
    "using machine learning."
)

# Add space between the introduction and the KPI section
st.write("")

# Create three equally sized columns
col1, col2, col3 = st.columns(3)

# Show the number of hourly observations in the dataset
with col1:
    st.metric(
        label="Hourly observations",
        value="8,760"
    )

# Show the Random Forest model's test R-squared score
with col2:
    st.metric(
        label="Model R² (test data)",
        value="0.73"
    )

# Show the model's mean absolute error on the test dataset
with col3:
    st.metric(
        label="MAE (bikes/hour)",
        value="219.5",
        help="Average prediction error, measured in rented bikes per hour."
    )


# Import pandas to work with the bike rental dataset
import pandas as pd

# Load the enriched dataset from the data folder
df = pd.read_csv("data/bikeflow_enriched.csv")

# Convert dates using the YYYY-MM-DD format
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

# Create a binary feature: 0 = weekday, 1 = weekend
df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)

# Keep only hours when the bike rental system was operating
df_operating = df[df["Functioning Day"] == "Yes"]

# Calculate the average number of rented bikes for each hour
hourly_demand = (
    df_operating.groupby("Hour")["Rented Bike Count"]
    .mean()
    .reset_index()
)

# Import Altair to create a customized chart
import altair as alt

# Introduce the historical rental patterns
st.subheader("Explore rental patterns")

#Expalin the time perido and scope of the chart
st.caption(
    "Average hourly rentals accross all months, "
    "December 2017 - November 2018. Operation hours only."
)

# Let users choose which type of day to explore
day_filter = st.segmented_control(
    "Compare rental patterns",
    options=["All days", "Weekdays", "Weekends"],
    default="All days"
)

# Filter the data based on the selected day type
if day_filter == "Weekdays":
    filtered_df = df_operating[df_operating["is_weekend"] == 0]

elif day_filter == "Weekends":
    filtered_df = df_operating[df_operating["is_weekend"] == 1]

else:
    filtered_df = df_operating

# Calculate average rentals per hour for the selected days
hourly_demand = (
    filtered_df.groupby("Hour")["Rented Bike Count"]
    .mean()
    .reset_index()
)
# Display a headline based on the selected day type
if day_filter == "Weekdays":
    st.subheader("Weekday rentals show morning and evening peaks")

elif day_filter == "Weekends":
    st.subheader("Weekend rentals build gradually throughout the day")

else:
    st.subheader("Bike rentals peak in the early evening")

# Create a line chart showing average hourly rentals
chart = (
    alt.Chart(hourly_demand)
    .mark_line(
        color="#65DED2",  # Use the dashboard's accent color
        strokeWidth=3      # Make the line easy to see
    )
    .encode(
        # Configure the horizontal axis
        x=alt.X(
            "Hour:Q",
            title="Hour of day",
            scale=alt.Scale(domain=[0, 23], nice=False),
            axis=alt.Axis(
            values=list(range(0, 24, 2)),
            grid=False,
            labelColor="#D0E2E9",
            titleColor="#D0E2E9"
        )
),

        # Configure the vertical axis with a consistent scale
        y=alt.Y(
            "Rented Bike Count:Q",
            title="Average rented bikes",
            scale=alt.Scale(domain=[0, 2200], nice=False),
            axis=alt.Axis(
            grid=True,
            gridColor="#355063",
            gridOpacity=0.4,
            labelColor="#D0E2E9",
            titleColor="#D0E2E9"
        )
),

        # Show rounded values when hovering over the line
        tooltip=[
            alt.Tooltip("Hour:Q", title="Hour"),
            alt.Tooltip(
                "Rented Bike Count:Q",
                title="Average rentals",
                format=",.0f"
            )
        ]
    )
    
# Set chart height and remove unnecessary visual borders
.properties(height=350)
.configure_view(
    stroke=None  # Remove the border around the chart
)
.configure_axis(
    gridColor="#294356",  # Use subtle gridlines
    labelColor="#C9D6DE",  # Use soft text colors
    titleColor="#C9D6DE"
)
.configure(
    background="#102B40"  # Match the dashboard background
)

)

# Display the chart across the available page width
st.altair_chart(chart, width="stretch")

# -----------------------------------------
# PREDICT BIKE DEMAND
# -----------------------------------------

# Cache the trained model to avoid loading it repeatedly
@st.cache_resource
def load_model():
    # Load the saved Random Forest pipeline
    return joblib.load("bikeflow_model.joblib")

# Call the function and store the trained model
model = load_model()

# Create a new section for predictions
st.divider()
st.header("Predict bike demand")


# Introduce the prediction section
st.write(
    "Choose a scenario to explore hourly bike rentals in Seoul."
)

# Create two columns for a clean and structured layout
col_time, col_weather = st.columns(2)

# -----------------------------------------
# TIME AND CALENDAR FEATURES
# -----------------------------------------

with col_time:

    st.subheader("Time & calendar")

    # Select the hour of the day
    hour = st.slider("Hour of day", 0, 23, 18)

    # Create a dictionary mapping month numbers to month names
    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    # Display month names while retaining numerical values
    month = st.selectbox(
        "Month",
        options=list(month_names.keys()),
        index=5,
        format_func=lambda x: month_names[x]
    )

    # Select weekday or weekend
    day_type = st.radio(
        "Day type",
        ["Weekday", "Weekend"],
        horizontal=True
    )

    # Convert the selected day type to a binary feature
    is_weekend = int(day_type == "Weekend")

    # Let the user indicate whether it is a public holiday
    public_holiday = st.checkbox("Public holiday")

    # Convert the checkbox to the original dataset categories
    holiday = "Holiday" if public_holiday else "No Holiday"


# -----------------------------------------
# WEATHER FEATURES
# -----------------------------------------

with col_weather:

    st.subheader("Weather conditions")

    # Select temperature in degrees Celsius
    temperature = st.slider(
        "Temperature (°C)", -17.8, 39.4, 13.5, 0.1
    )

    # Select rainfall in millimetres
    rainfall = st.slider(
        "Rainfall (mm)", 0.0, 35.0, 0.0, 0.5
    )

    # Hide additional weather settings to reduce visual clutter
    with st.expander("Advanced weather settings"):

        # Select relative humidity
        humidity = st.slider(
            "Humidity (%)", 1, 98, 57
        )

        # Select wind speed in metres per second
        wind_speed = st.slider(
            "Wind speed (m/s)", 0.0, 7.4, 1.5, 0.1
        )

        # Select snowfall in centimetres
        snowfall = st.slider(
            "Snowfall (cm)", 0.0, 8.8, 0.0, 0.1
        )

        # Select solar radiation
        solar_radiation = st.slider(
            "Solar radiation (MJ/m²)", 0.0, 3.52, 0.01, 0.01
        )

# Create a DataFrame containing the user's selected values
input_data = pd.DataFrame({
    "Hour": [hour],
    "month": [month],
    "is_weekend": [is_weekend],
    "Temperature(°C)": [temperature],
    "Humidity(%)": [humidity],
    "Wind speed (m/s)": [wind_speed],
    "Rainfall(mm)": [rainfall],
    "Snowfall (cm)": [snowfall],
    "Solar Radiation (MJ/m2)": [solar_radiation],
    "Holiday": [holiday]
})

# Create a button to generate a prediction
if st.button("Predict demand", type="primary"):

    # Pass the user's input to the trained model
    prediction = model.predict(input_data)

    # Extract the first prediction and round to a whole number
    predicted_bikes = round(prediction[0])

    # Display the prediction using HTML instead of Markdown
    st.html(
        f"""
        <div style="
            background-color: #1E4054;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        ">
            <div style="color: #D0E2E9; font-size: 14px;">
                PREDICTED HOURLY BIKE RENTALS
            </div>

            <div style="
                color: #65DED2;
                font-size: 64px;
                font-weight: 600;
            ">
                {predicted_bikes:,}
            </div>

            <div style="color: #FFFFFF; font-size: 16px;">
                Estimated rentals within one hour
            </div>
        </div>
        """
    )

    # Explain the scope of the prediction
    st.caption(
        "Estimated hourly rentals based on historical "
        "Seoul bike rental data from 2017–2018."
    )

# Separate the prediction section from the model information
st.divider()

# Introduce the model information section
st.header("Behind the prediction")

st.write(
    "Understand how the model works and how to interpret its predictions."
)

# Explain Random Forest to a non-technical audience
with st.expander("How does the model work?"):
    st.write(
        "The Random Forest model combines predictions from 100 decision "
        "trees. Each tree learns patterns from historical bike rental data. "
        "By averaging their predictions, the model estimates hourly bike "
        "rentals based on time, calendar and weather conditions."
    )

# Explain model performance using the held-out test results
with st.expander("How accurate is the model?"):
    st.write(
        "The model explains approximately 73% of the variation in hourly "
        "bike rentals in the test data (R² = 0.73). "
        "On average, its predictions differ from actual rentals by "
        "approximately 220 bikes per hour (MAE)."
    )

# Communicate the limitations of the model
with st.expander("What are the limitations?"):
    st.write(
        "The model is based on historical data from Seoul collected "
        "between December 2017 and November 2018. "
        "It may not reflect current travel patterns or changes in "
        "cycling infrastructure."
    )

    st.write(
        "The model also tends to underestimate rentals during periods "
        "of particularly high demand. Predictions should therefore "
        "be interpreted as estimates rather than exact forecasts."
    )

    st.write(
        "The model assumes that the bike rental system is operational "
        "and predicts recorded rentals rather than unmet demand."
    )

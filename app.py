
# Import Streamlit to build the web application
import streamlit as st

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
        font-size: 3.5rem;
        font-weight: 700;
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
    </style>
    """,
    unsafe_allow_html=True
)

# Display the application name
st.title("BikeFlow Seoul")

# Display a panoramic photograph of Seoul
st.image(
    "assets/seoul_night.jpg",
    width="stretch"
)

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
        label="Model R²",
        value="0.73"
    )

# Show the model's mean absolute error on the test dataset
with col3:
    st.metric(
        label="Mean absolute error",
        value="219.5",
        help="Average prediction error, measured in rented bikes per hour."
    )


import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

def search_and_display_rollercoasters(df):
    # Function to filter rollercoasters based on user input
    def filter_rollercoasters(query):
        return df[df.apply(lambda row: any(query.lower() in str(cell).lower() for cell in row), axis=1)]

    def display_rollercoaster_info(selected_coaster):
        st.subheader("Rollercoaster Information")
        for key, value in selected_coaster.items():
            if pd.notna(value):
                st.write(f"**{key.capitalize()}:** {value}")

    def display_rollercoaster_image(coaster):
        img_url = coaster.get('Image_URL')
        if img_url:
            response = requests.get(img_url)
            if response.status_code == 200:
                img_data = response.content
                img = Image.open(BytesIO(img_data))
                img = img.resize((300, 200), Image.ANTIALIAS)
                st.image(img, caption=coaster.get('Name'), use_column_width=True)
            else:
                st.write("Image not available.")

    st.title("🎢 Rollercoaster Search")

    # Search query input
    query = st.text_input("Enter rollercoaster name or amusement park:", "")

    # Filter rollercoasters based on query
    if query:
        results = filter_rollercoasters(query)
        if results.empty:
            st.info("No rollercoasters found matching the search query.")
        else:
            st.subheader("🔍 Rollercoasters Found")
            # Display all related rollercoasters in a list
            for i, (_, rollercoaster) in enumerate(results.iterrows(), start=1):
                # Create an expander for each rollercoaster
                with st.expander(f"{rollercoaster['Name']} - {rollercoaster['Amusement Park']}"):
                    # Display rollercoaster information and image
                    display_rollercoaster_info(rollercoaster)
                    display_rollercoaster_image(rollercoaster)

# Example usage:
# Assuming df is your Pandas DataFrame containing rollercoaster information
# Call the function passing the DataFrame as argument
if __name__ == '__main__':
    df = pd.read_csv('data.csv')  # Assuming 'data.csv' is the CSV file containing rollercoaster information
    search_and_display_rollercoasters(df)
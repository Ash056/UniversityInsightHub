import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="University Survey Analytics",
    page_icon="🎓",
    layout="wide"
)

def main():
    st.title("🎓 University Survey Analytics Tool")
    st.write("Testing basic functionality...")
    
    # Simple login form
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.subheader("Please log in to access the system")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "dean" and password == "dean123":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success(f"Welcome, {st.session_state.username}!")
        st.write("Authentication successful - the system is working!")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    main()
import streamlit as st
import hashlib

# Simple role-based access control
USERS = {
    "dean": {
        "password": "dean123",
        "role": "Dean"
    },
    "admin": {
        "password": "admin123", 
        "role": "Administrator"
    },
    "analyst": {
        "password": "analyst123",
        "role": "Analyst"
    },
    "viewer": {
        "password": "viewer123",
        "role": "Viewer"
    }
}

# Role permissions mapping
PERMISSIONS = {
    "Dean": [
        "upload_data", "view_eda", "view_nlp", "view_topics", 
        "view_sentiment", "view_advanced", "export_data", "manage_users"
    ],
    "Administrator": [
        "upload_data", "view_eda", "view_nlp", "view_topics", 
        "view_sentiment", "view_advanced", "export_data"
    ],
    "Analyst": [
        "view_eda", "view_nlp", "view_topics", "view_sentiment", "view_advanced"
    ],
    "Viewer": [
        "view_eda", "view_sentiment"
    ]
}

def hash_password(password):
    """Simple password hashing for demo purposes"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password, role):
    """Authenticate user and validate role"""
    if username in USERS:
        user_data = USERS[username]
        if user_data["password"] == password and user_data["role"] == role:
            return True
    return False

def check_permissions(user_role, permission):
    """Check if user role has specific permission"""
    if user_role not in PERMISSIONS:
        return False
    return permission in PERMISSIONS[user_role]

def get_user_permissions(user_role):
    """Get all permissions for a user role"""
    return PERMISSIONS.get(user_role, [])

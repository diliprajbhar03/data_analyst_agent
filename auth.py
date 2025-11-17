import os
import streamlit as st
from functools import wraps

def get_secret_or_env(key, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

def require_password(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            password = st.sidebar.text_input("?? Enter Password", type="password")

            correct_password = (
                get_secret_or_env("app_password") or
                get_secret_or_env("APP_PASSWORD") or
                "mysecret"
            )

            if password == correct_password:
                st.session_state["authenticated"] = True
                st.rerun()
            elif password:
                st.error("Incorrect password.")
                st.stop()
            else:
                st.stop()

        return func(*args, **kwargs)

    return wrapper

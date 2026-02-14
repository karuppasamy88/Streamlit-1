"""Simple Streamlit app with safe input handling."""

from __future__ import annotations

import streamlit as st

from calculator import calculate_percentage_change


def main() -> None:
    st.set_page_config(page_title="Percentage Change Calculator", page_icon="📈")
    st.title("📈 Percentage Change Calculator")
    st.write("Enter two values to calculate the percentage change.")

    original = st.number_input("Original value", value=100.0, format="%.2f")
    updated = st.number_input("Updated value", value=110.0, format="%.2f")

    if st.button("Calculate"):
        try:
            result = calculate_percentage_change(original, updated)
            st.success(f"Percentage change: {result:.2f}%")
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()

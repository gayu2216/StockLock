import streamlit as st

main = st.Page(
    "pages/0_main.py",
    title="🏡 Home Page",
)

dca = st.Page(
    "pages/1_dca.py",
    title="💵 Dollar-Cost Averaging",
)

swp = st.Page(
    "pages/2_swp.py",
    title="🔄 Systematic Withdrawal Plan",
)

lumpsum = st.Page(
    "pages/3_lumpsum.py",
    title="💰 Lump Sum Investment",
)

neural = st.Page(
    "pages/4_stocks.py",
    title="📊 Stock Analysis",
)

chat = st.Page(
    "pages/5_chat.py",
    title="🤖 Chatbot",
)


pg = st.navigation(pages=[main, dca, swp, lumpsum, neural,chat])
pg.run()

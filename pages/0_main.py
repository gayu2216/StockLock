import streamlit as st

# ----------- SESSION SETUP ----------- 
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = None

# ----------- NAVIGATION ----------- 
def go_to(page_name):
    st.session_state.page = page_name

# ----------- MAIN PAGE SWITCHING ----------- 
def main():
    # Add custom CSS for pink background and styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFDBED, #ffffff);
            background-size: cover;
        }
        
        /* Styling for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 6px;
            color: #4B8BBE;
            font-weight: 600;
            padding: 10px 20px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4B8BBE, #5A98D1);
            color: white !important;
        }
        
        /* Card styling for features */
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Sidebar styling */
        [data-testid=stSidebar] {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1rem;
        }
        
        /* Navigation button styling */
        .nav-button {
            background: linear-gradient(135deg, #4B8BBE, #5A98D1);
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 5px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(75, 139, 190, 0.4);
        }
        
        /* Navigation container */
        .nav-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add logo to sidebar
    with st.sidebar:
        st.image("/Users/gayathriutla/Desktop/Projects/Final_bot/logo.jpeg", width=150)
        st.markdown("<h3 style='text-align: center; color: #4B8BBE;'>Stock lock</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your investment buddy 🤝</p>", unsafe_allow_html=True)
    
    # Create centered navigation menu
    st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
    
    # Create navigation buttons using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            go_to("home")
            
    with col2:
        if st.button("🪙 DCA", key="nav_dca", use_container_width=True):
            st.switch_page("/Users/gayathriutla/Desktop/Projects/Final_bot/pages/1_dca.py")
            
    with col3:
        if st.button("🔄 SWP", key="nav_swp", use_container_width=True):
            st.switch_page("/Users/gayathriutla/Desktop/Projects/Final_bot/pages/2_swp.py")
            
    with col4:
        if st.button("💰 Lump Sum", key="nav_lump", use_container_width=True):
            st.switch_page("/Users/gayathriutla/Desktop/Projects/Final_bot/pages/3_lumpsum.py")
            
    with col5:
        if st.button("📈 Stocks", key="nav_stocks", use_container_width=True):
            st.switch_page("/Users/gayathriutla/Desktop/Projects/Final_bot/pages/4_stocks.py")
    
    # Create second row for Chat button
    col_space1, col_chat, col_space2 = st.columns([2, 1, 2])
    with col_chat:
        if st.button("🧠 Chat with AI", key="nav_chat", use_container_width=True):
            st.switch_page("/Users/gayathriutla/Desktop/Projects/Final_bot/pages/5_chat.py")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area (Home page)
    if st.session_state.page == "home":
        st.markdown("<h1 style='color:#4B8BBE; text-align:center;'>Welcome to Stock lock</h1>", unsafe_allow_html=True)
        
        # App features summary
        st.markdown("<h2 style='color:#4B8BBE;'>Our App Features</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style='color:#4B8BBE;'>🪙 Dollar-Cost Averaging (DCA)</h3>
            <p>Invest fixed amounts at regular intervals regardless of market prices. This strategy helps reduce the impact of volatility and is perfect for beginners or long-term investors who want to build wealth consistently over time.</p>
        </div>
        
        <div class="feature-card">
            <h3 style='color:#4B8BBE;'>🔄 Systematic Withdrawal Plan (SWP)</h3>
            <p>Withdraw predetermined amounts from your investments at regular intervals. This approach provides a steady income stream while allowing the remaining investment to potentially continue growing.</p>
        </div>
        
        <div class="feature-card">  
            <h3 style='color:#4B8BBE;'>💰 Lump Sum Investment</h3>
            <p>Invest a large amount all at once. This approach can be beneficial when you believe the market is undervalued or when you receive a significant windfall such as a bonus or inheritance.</p>
        </div>
        
        <div class="feature-card">
            <h3 style='color:#4B8BBE;'>📈 Stock Investing</h3>
            <p>Buy shares in individual companies and build a portfolio. Our tools help you analyze stocks, understand their potential, and make informed decisions based on your investment goals and risk tolerance.</p>
        </div>
        
        <div class="feature-card">
            <h3 style='color:#4B8BBE;'>🧠 Chat with Sherlock AI</h3>
            <p>Get personalized investment advice, explanations of financial concepts, and answers to your questions from our AI assistant. Sherlock is here to help you navigate the complex world of investing with confidence.</p>
        </div>
        """, unsafe_allow_html=True)

# Run the main function to display the correct page
main()
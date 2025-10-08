sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis","Strength vs Weakness","Match by Match Analysis")
)
if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())

    # Filter the data for the selected player
    player_info = idf[idf['batsman'] == player_name].iloc[0]
    # Tabs for "Overview", "Career Statistics", and "Current Form"
    tab1, tab2 = st.tabs(["Career Statistics", "Current Form"])
    with tab1:
            st.header("Career Statistics")
    
            # Dropdown for Batting or Bowling selection
            option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))
    
            # Show Career Averages based on the dropdown
            st.subheader("Career Performance")
    
            # Display Career Averages based on selection
            if option == "Batting":
                # Create a temporary DataFrame and filter the player's row
                temp_df = idf.drop(columns=['final_year'])
                player_stats = temp_df[temp_df['batsman'] == player_name]  # Filter for the selected player
    
                # Convert column names to uppercase and replace underscores with spaces
                player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]
                # player_stats=round_up_floats(player_stats)
                # Display the player's statistics in a table format with bold headers
                st.markdown("### Batting Statistics")
                columns_to_convert = ['RUNS','HUNDREDS', 'FIFTIES','THIRTIES', 'HIGHEST SCORE']
    
                   # Fill NaN values with 0
                player_stats[columns_to_convert] = player_stats[columns_to_convert].fillna(0)
                    
                   # Convert the specified columns to integer type
                player_stats[columns_to_convert] = player_stats[columns_to_convert].astype(int)
            
                st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))                
               

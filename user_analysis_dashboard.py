#!/usr/bin/env python3
"""
User Demographics and Behavior Analysis Dashboard
Analyzes Data Axle enriched user data to provide insights for business teams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(filename='data_axle_results.csv'):
    """Load and perform basic cleaning of the data"""
    print(f"Loading data from {filename}...")
    
    # Read with specific columns we need for analysis
    columns_to_read = [
        'email', 'data.document.attributes.first_name', 'data.document.attributes.last_name',
        'data.document.attributes.gender', 'data.document.attributes.city', 
        'data.document.attributes.state', 'data.document.attributes.postal_code',
        'data.document.attributes.family.estimated_income', 'data.document.attributes.family.estimated_wealth[0]',
        'data.document.attributes.family.estimated_education_level', 'data.document.attributes.family.home_owner',
        'data.document.attributes.family.adult_count', 'data.document.attributes.family.member_count',
        'data.document.attributes.estimated_married', 'data.document.attributes.lifestyle_segment',
        'data.document.attributes.political_party_affiliation'
    ]
    
    # Interest columns (sample - there are many more)
    exclude_cols = [
        'data.document.attributes.interests.internet', 'data.document.attributes.interests.credit_cards',
        'data.document.attributes.interests.catalogs'
    ]

    all_columns = pd.read_csv(filename, nrows=0).columns.tolist()
    use_cols = [col for col in all_columns if col not in exclude_cols]
    
    try:
        # Try to read specific columns first
        df = pd.read_csv(filename, usecols=use_cols, low_memory=False)
    except:
        # If that fails, read all columns
        print("Reading all columns...")
        df = pd.read_csv(filename, low_memory=False)
    
    print(f"Loaded {len(df)} records")
    return df

def create_geographic_analysis(df):
    """Create geographic distribution visualizations"""
    print("Creating geographic analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Geographic Distribution of Users', fontsize=16, fontweight='bold')
    
    # State distribution
    if 'data.document.attributes.state' in df.columns:
        state_counts = df['data.document.attributes.state'].value_counts().head(15)
        axes[0,0].bar(range(len(state_counts)), state_counts.values)
        axes[0,0].set_xticks(range(len(state_counts)))
        axes[0,0].set_xticklabels(state_counts.index, rotation=45)
        axes[0,0].set_title('Top 15 States by User Count')
        axes[0,0].set_ylabel('Number of Users')
    
    # City distribution (top cities)
    if 'data.document.attributes.city' in df.columns:
        city_counts = df['data.document.attributes.city'].value_counts().head(15)
        axes[0,1].barh(range(len(city_counts)), city_counts.values)
        axes[0,1].set_yticks(range(len(city_counts)))
        axes[0,1].set_yticklabels(city_counts.index)
        axes[0,1].set_title('Top 15 Cities by User Count')
        axes[0,1].set_xlabel('Number of Users')
    
    # Geographic heat map by state (if we have enough data)
    if 'data.document.attributes.state' in df.columns:
        state_data = df['data.document.attributes.state'].value_counts()
        # Create a simple visualization
        top_states = state_data.head(20)
        bars = axes[1,0].bar(range(len(top_states)), top_states.values, 
                            color=plt.cm.viridis(np.linspace(0, 1, len(top_states))))
        axes[1,0].set_xticks(range(len(top_states)))
        axes[1,0].set_xticklabels(top_states.index, rotation=45)
        axes[1,0].set_title('User Concentration by State (Top 20)')
        axes[1,0].set_ylabel('Number of Users')
    
    # Geographic diversity metrics
    if 'data.document.attributes.state' in df.columns:
        total_states = df['data.document.attributes.state'].nunique()
        total_cities = df['data.document.attributes.city'].nunique() if 'data.document.attributes.city' in df.columns else 0
        
        metrics_text = f"""Geographic Coverage:
        
Total States: {total_states}
Total Cities: {total_cities}
Most Common State: {df['data.document.attributes.state'].mode().iloc[0] if len(df['data.document.attributes.state'].mode()) > 0 else 'N/A'}
        
Top 3 States:
"""
        if len(state_counts) >= 3:
            for i, (state, count) in enumerate(state_counts.head(3).items()):
                pct = (count / len(df)) * 100
                metrics_text += f"{i+1}. {state}: {count} ({pct:.1f}%)\n"
        
        axes[1,1].text(0.1, 0.9, metrics_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Geographic Summary')
    
    plt.tight_layout()
    plt.savefig('geographic_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_demographic_analysis(df):
    """Create demographic distribution visualizations"""
    print("Creating demographic analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('User Demographics Analysis', fontsize=16, fontweight='bold')
    
    # Gender distribution
    if 'data.document.attributes.gender' in df.columns:
        gender_counts = df['data.document.attributes.gender'].value_counts()
        axes[0,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Gender Distribution')
    
    # Marital status
    if 'data.document.attributes.estimated_married' in df.columns:
        married_counts = df['data.document.attributes.estimated_married'].value_counts()
        axes[0,1].pie(married_counts.values, labels=['Married' if x else 'Single' for x in married_counts.index], 
                     autopct='%1.1f%%')
        axes[0,1].set_title('Marital Status Distribution')
    
    # Home ownership
    if 'data.document.attributes.family.home_owner' in df.columns:
        home_counts = df['data.document.attributes.family.home_owner'].value_counts()
        axes[0,2].pie(home_counts.values, labels=['Owner' if x else 'Renter' for x in home_counts.index], 
                     autopct='%1.1f%%')
        axes[0,2].set_title('Home Ownership')
    
    # Education level
    if 'data.document.attributes.family.estimated_education_level' in df.columns:
        edu_counts = df['data.document.attributes.family.estimated_education_level'].value_counts()
        axes[1,0].bar(range(len(edu_counts)), edu_counts.values)
        axes[1,0].set_xticks(range(len(edu_counts)))
        axes[1,0].set_xticklabels(edu_counts.index, rotation=45)
        axes[1,0].set_title('Education Level Distribution')
        axes[1,0].set_ylabel('Number of Users')
    
    # Family size
    if 'data.document.attributes.family.member_count' in df.columns:
        family_size = df['data.document.attributes.family.member_count'].value_counts().sort_index()
        axes[1,1].bar(family_size.index, family_size.values)
        axes[1,1].set_title('Family Size Distribution')
        axes[1,1].set_xlabel('Family Members')
        axes[1,1].set_ylabel('Number of Users')
    
    # Political affiliation
    if 'data.document.attributes.political_party_affiliation' in df.columns:
        pol_counts = df['data.document.attributes.political_party_affiliation'].value_counts()
        if len(pol_counts) > 0:
            axes[1,2].pie(pol_counts.values, labels=pol_counts.index, autopct='%1.1f%%')
            axes[1,2].set_title('Political Affiliation')
        else:
            axes[1,2].text(0.5, 0.5, 'No Political Data Available', 
                          transform=axes[1,2].transAxes, ha='center', va='center')
            axes[1,2].set_title('Political Affiliation')
    
    plt.tight_layout()
    plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_financial_analysis(df):
    """Create financial/income distribution visualizations"""
    print("Creating financial analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Financial Profile of Users', fontsize=16, fontweight='bold')
    
    # Income distribution
    if 'data.document.attributes.family.estimated_income' in df.columns:
        income_data = df['data.document.attributes.family.estimated_income'].dropna()
        if len(income_data) > 0:
            axes[0,0].hist(income_data, bins=30, edgecolor='black', alpha=0.7)
            axes[0,0].set_title('Estimated Income Distribution')
            axes[0,0].set_xlabel('Income ($)')
            axes[0,0].set_ylabel('Number of Users')
            
            # Add income statistics
            income_stats = f"""Income Statistics:
Mean: ${income_data.mean():,.0f}
Median: ${income_data.median():,.0f}
Q1: ${income_data.quantile(0.25):,.0f}
Q3: ${income_data.quantile(0.75):,.0f}"""
            axes[0,0].text(0.7, 0.95, income_stats, transform=axes[0,0].transAxes, 
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Wealth distribution
    if 'data.document.attributes.family.estimated_wealth[0]' in df.columns:
        wealth_data = df['data.document.attributes.family.estimated_wealth[0]'].dropna()
        if len(wealth_data) > 0:
            axes[0,1].hist(wealth_data, bins=30, edgecolor='black', alpha=0.7, color='green')
            axes[0,1].set_title('Estimated Wealth Distribution')
            axes[0,1].set_xlabel('Wealth ($)')
            axes[0,1].set_ylabel('Number of Users')
    
    # Income vs Home Ownership
    if 'data.document.attributes.family.estimated_income' in df.columns and 'data.document.attributes.family.home_owner' in df.columns:
        income_by_ownership = df.groupby('data.document.attributes.family.home_owner')['data.document.attributes.family.estimated_income'].mean()
        if len(income_by_ownership) > 0:
            bars = axes[1,0].bar(['Renter', 'Owner'], income_by_ownership.values)
            axes[1,0].set_title('Average Income by Home Ownership')
            axes[1,0].set_ylabel('Average Income ($)')
            
            # Add value labels on bars
            for bar, value in zip(bars, income_by_ownership.values):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                              f'${value:,.0f}', ha='center', va='bottom')
    
    # Income brackets
    if 'data.document.attributes.family.estimated_income' in df.columns:
        income_data = df['data.document.attributes.family.estimated_income'].dropna()
        if len(income_data) > 0:
            # Create income brackets
            income_brackets = pd.cut(income_data, 
                                   bins=[0, 25000, 50000, 75000, 100000, 150000, float('inf')],
                                   labels=['<$25K', '$25K-$50K', '$50K-$75K', '$75K-$100K', '$100K-$150K', '>$150K'])
            bracket_counts = income_brackets.value_counts()
            
            axes[1,1].pie(bracket_counts.values, labels=bracket_counts.index, autopct='%1.1f%%')
            axes[1,1].set_title('Income Bracket Distribution')
    
    plt.tight_layout()
    plt.savefig('financial_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_interests_analysis(df):
    """Create interests and lifestyle analysis"""
    print("Creating interests analysis...")

    # Find interest columns
    interest_cols = [col for col in df.columns if 'interests.' in col and col not in ['data.document.attributes.interests.id', 'data.document.attributes.interests.created_at']]
    
    if len(interest_cols) == 0:
        print("No interest data found in the dataset")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    fig.suptitle('User Interests and Lifestyle Analysis\n(Scores: 1=Low Interest, 9=High Interest)', fontsize=16, fontweight='bold')
    
    # Analyze interest scores properly (1-9 scale)
    interest_analysis = {}
    for col in interest_cols[:100]:
        interest_name = col.split('.')[-1].replace('_', ' ').title()
        interest_data = df[col].dropna()
        
        if len(interest_data) > 0:
            # Convert to numeric, handling any string values
            try:
                interest_data = pd.to_numeric(interest_data, errors='coerce').dropna()
                if len(interest_data) > 0:
                    interest_analysis[interest_name] = {
                        'user_count': len(interest_data),
                        'avg_score': interest_data.mean(),
                        'high_interest_users': (interest_data >= 7).sum(),  # Users with strong interest (7-9)
                        'weighted_score': len(interest_data) * interest_data.mean()  # Volume * Intensity
                    }
            except:
                continue
    
    if interest_analysis:
        # Top interests by user volume
        top_by_volume = sorted(interest_analysis.items(), key=lambda x: x[1]['user_count'], reverse=True)[:15]
        interests, data = zip(*top_by_volume)
        user_counts = [d['user_count'] for d in data]
        
        axes[0,0].barh(range(len(interests)), user_counts, color='skyblue')
        axes[0,0].set_yticks(range(len(interests)))
        axes[0,0].set_yticklabels(interests)
        axes[0,0].set_title('Top 15 Interests by User Volume')
        axes[0,0].set_xlabel('Number of Users with This Interest')
        
        # Top interests by average score (intensity)
        top_by_intensity = sorted(interest_analysis.items(), key=lambda x: x[1]['avg_score'], reverse=True)[:15]
        interests_int, data_int = zip(*top_by_intensity)
        avg_scores = [d['avg_score'] for d in data_int]
        
        bars = axes[0,1].barh(range(len(interests_int)), avg_scores, color='lightcoral')
        axes[0,1].set_yticks(range(len(interests_int)))
        axes[0,1].set_yticklabels(interests_int)
        axes[0,1].set_title('Top 15 Interests by Average Score (Intensity)')
        axes[0,1].set_xlabel('Average Interest Score (1-9)')
        axes[0,1].set_xlim(0, 9)
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, avg_scores)):
            axes[0,1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                          f'{score:.1f}', ha='left', va='center', fontsize=8)
        
        # High engagement interests (users with scores 7-9)
        high_engagement = sorted(interest_analysis.items(), key=lambda x: x[1]['high_interest_users'], reverse=True)[:15]
        interests_he, data_he = zip(*high_engagement)
        high_users = [d['high_interest_users'] for d in data_he]
        
        axes[0,2].barh(range(len(interests_he)), high_users, color='lightgreen')
        axes[0,2].set_yticks(range(len(interests_he)))
        axes[0,2].set_yticklabels(interests_he)
        axes[0,2].set_title('Top 15 Interests by High Engagement\n(Users with Scores 7-9)')
        axes[0,2].set_xlabel('Number of Highly Engaged Users')
    
    # Lifestyle segments
    if 'data.document.attributes.lifestyle_segment' in df.columns:
        lifestyle_counts = df['data.document.attributes.lifestyle_segment'].value_counts().head(10)
        if len(lifestyle_counts) > 0:
            axes[1,0].pie(lifestyle_counts.values, labels=lifestyle_counts.index, autopct='%1.1f%%')
            axes[1,0].set_title('Top Lifestyle Segments')
    
    # Interest score distribution
    if interest_analysis:
        all_scores = []
        for col in interest_cols[:50]:  # Sample from interest columns
            scores = pd.to_numeric(df[col], errors='coerce').dropna()
            all_scores.extend(scores.tolist())
        
        if all_scores:
            axes[1,1].hist(all_scores, bins=range(1, 11), edgecolor='black', alpha=0.7, color='orange')
            axes[1,1].set_title('Distribution of Interest Scores\n(All Interests Combined)')
            axes[1,1].set_xlabel('Interest Score (1-9)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_xticks(range(1, 10))
            
            # Add statistics
            mean_score = np.mean(all_scores)
            axes[1,1].axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                             label=f'Mean: {mean_score:.1f}')
            axes[1,1].legend()
    
    # Interest insights and statistics
    total_users = len(df)
    total_interest_categories = len(interest_analysis)
    
    if interest_analysis:
        # Calculate average scores across all interests
        all_avg_scores = [data['avg_score'] for data in interest_analysis.values()]
        overall_avg_score = np.mean(all_avg_scores)
        
        # Find most engaging interest
        most_engaging = max(interest_analysis.items(), key=lambda x: x[1]['avg_score'])
        most_popular = max(interest_analysis.items(), key=lambda x: x[1]['user_count'])
        
        summary_text = f"""Interest Insights:

📊 OVERVIEW:
• Total Users: {total_users:,}
• Interest Categories: {total_interest_categories}
• Overall Avg Score: {overall_avg_score:.1f}/9

🔥 HIGHEST ENGAGEMENT:
• {most_engaging[0]}: {most_engaging[1]['avg_score']:.1f}/9
• {most_engaging[1]['user_count']:,} users

👥 MOST POPULAR:
• {most_popular[0]}: {most_popular[1]['user_count']:,} users
• Avg Score: {most_popular[1]['avg_score']:.1f}/9

📈 HIGH INTEREST USERS (7-9):"""
        
        for i, (interest, data) in enumerate(high_engagement[:3]):
            pct = (data['high_interest_users'] / total_users) * 100
            summary_text += f"\n{i+1}. {interest}: {data['high_interest_users']:,} ({pct:.1f}%)"
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Key Interest Insights')
    
    plt.tight_layout()
    plt.savefig('interests_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_summary_dashboard(df):
    """Create a high-level summary dashboard"""
    print("Creating summary dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('User Base Summary Dashboard', fontsize=18, fontweight='bold')
    
    # Key metrics
    total_users = len(df)
    unique_states = df['data.document.attributes.state'].nunique() if 'data.document.attributes.state' in df.columns else 0
    unique_cities = df['data.document.attributes.city'].nunique() if 'data.document.attributes.city' in df.columns else 0
    avg_income = df['data.document.attributes.family.estimated_income'].mean() if 'data.document.attributes.family.estimated_income' in df.columns else 0
    
    # User volume by state (top 10)
    if 'data.document.attributes.state' in df.columns:
        state_counts = df['data.document.attributes.state'].value_counts().head(10)
        axes[0,0].bar(range(len(state_counts)), state_counts.values, color='skyblue')
        axes[0,0].set_xticks(range(len(state_counts)))
        axes[0,0].set_xticklabels(state_counts.index, rotation=45)
        axes[0,0].set_title('Top 10 States by User Volume')
        axes[0,0].set_ylabel('Number of Users')
    
    # Gender & Marital Status
    demo_data = []
    demo_labels = []
    
    if 'data.document.attributes.gender' in df.columns:
        gender_counts = df['data.document.attributes.gender'].value_counts()
        demo_data.extend(gender_counts.values)
        demo_labels.extend([f"{k} Gender" for k in gender_counts.index])
    
    if demo_data:
        axes[0,1].pie(demo_data, labels=demo_labels, autopct='%1.1f%%')
        axes[0,1].set_title('User Demographics')
    
    # Income distribution summary
    if 'data.document.attributes.family.estimated_income' in df.columns:
        income_data = df['data.document.attributes.family.estimated_income'].dropna()
        if len(income_data) > 0:
            income_brackets = pd.cut(income_data, 
                                   bins=[0, 30000, 60000, 100000, 150000, float('inf')],
                                   labels=['<$30K', '$30K-$60K', '$60K-$100K', '$100K-$150K', '>$150K'])
            bracket_counts = income_brackets.value_counts()
            
            axes[0,2].bar(range(len(bracket_counts)), bracket_counts.values, color='lightgreen')
            axes[0,2].set_xticks(range(len(bracket_counts)))
            axes[0,2].set_xticklabels(bracket_counts.index, rotation=45)
            axes[0,2].set_title('Income Distribution')
            axes[0,2].set_ylabel('Number of Users')
    
    # Key statistics text
    stats_text = f"""USER BASE OVERVIEW
    
📊 TOTAL USERS: {total_users:,}
    
🗺️ GEOGRAPHIC REACH:
• States: {unique_states}
• Cities: {unique_cities}
    
💰 FINANCIAL PROFILE:
• Avg Income: ${avg_income:,.0f}
    
👥 DEMOGRAPHICS:"""
    
    if 'data.document.attributes.gender' in df.columns:
        gender_dist = df['data.document.attributes.gender'].value_counts()
        for gender, count in gender_dist.items():
            pct = (count / total_users) * 100
            stats_text += f"\n• {gender}: {pct:.1f}%"
    
    if 'data.document.attributes.family.home_owner' in df.columns:
        homeowner_pct = (df['data.document.attributes.family.home_owner'].sum() / total_users) * 100
        stats_text += f"\n• Homeowners: {homeowner_pct:.1f}%"
    
    axes[1,0].text(0.05, 0.95, stats_text, transform=axes[1,0].transAxes, 
                  fontsize=12, verticalalignment='top', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1,0].set_xlim(0, 1)
    axes[1,0].set_ylim(0, 1)
    axes[1,0].axis('off')
    
    # Top cities
    if 'data.document.attributes.city' in df.columns:
        city_counts = df['data.document.attributes.city'].value_counts().head(10)
        axes[1,1].barh(range(len(city_counts)), city_counts.values, color='orange')
        axes[1,1].set_yticks(range(len(city_counts)))
        axes[1,1].set_yticklabels(city_counts.index)
        axes[1,1].set_title('Top 10 Cities')
        axes[1,1].set_xlabel('Number of Users')
    
    # Business insights
    insights_text = """KEY BUSINESS INSIGHTS
    
🎯 TARGET SEGMENTS:"""
    
    if 'data.document.attributes.state' in df.columns:
        top_state = df['data.document.attributes.state'].value_counts().index[0]
        top_state_pct = (df['data.document.attributes.state'].value_counts().iloc[0] / total_users) * 100
        insights_text += f"\n• {top_state}: {top_state_pct:.1f}% of users"
    
    if 'data.document.attributes.family.estimated_income' in df.columns:
        high_income = (df['data.document.attributes.family.estimated_income'] > 75000).sum()
        high_income_pct = (high_income / total_users) * 100
        insights_text += f"\n• High Income (>$75K): {high_income_pct:.1f}%"
    
    insights_text += f"""
    
📈 GROWTH OPPORTUNITIES:
• Geographic expansion potential
• Interest-based targeting
• Income-based product tiers
    
🔍 RECOMMENDATIONS:
• Focus marketing in top states
• Develop premium offerings
• Target high-income segments"""
    
    axes[1,2].text(0.05, 0.95, insights_text, transform=axes[1,2].transAxes, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_business_insights_report(df):
    """Generate key business insights about user interests"""
    print("Generating business insights report...")
    
    # Find interest columns
    interest_cols = [col for col in df.columns if 'interests.' in col and col not in ['data.document.attributes.interests.id', 'data.document.attributes.interests.created_at']]
    
    insights = {}
    for col in interest_cols:
        interest_name = col.split('.')[-1].replace('_', ' ').title()
        interest_data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(interest_data) > 0:
            insights[interest_name] = {
                'users': len(interest_data),
                'avg_score': interest_data.mean(),
                'high_interest': (interest_data >= 7).sum(),
                'moderate_interest': ((interest_data >= 4) & (interest_data < 7)).sum(),
                'low_interest': (interest_data < 4).sum()
            }
    
    # Business segments based on interests
    print("\n🎯 KEY BUSINESS INSIGHTS:")
    print("=" * 50)
    
    if insights:
        # High-value segments (high engagement + volume)
        high_value = sorted(insights.items(), 
                           key=lambda x: x[1]['high_interest'] * x[1]['avg_score'], 
                           reverse=True)[:5]
        
        print("\n📈 HIGH-VALUE INTEREST SEGMENTS:")
        for i, (interest, data) in enumerate(high_value):
            engagement_rate = (data['high_interest'] / data['users']) * 100
            print(f"{i+1}. {interest}:")
            print(f"   • {data['high_interest']:,} highly engaged users ({engagement_rate:.1f}%)")
            print(f"   • Average score: {data['avg_score']:.1f}/9")
            print(f"   • Total interested users: {data['users']:,}")
        
        # Emerging opportunities (moderate volume, high intensity)
        emerging = sorted([(k, v) for k, v in insights.items() if v['users'] >= 100 and v['avg_score'] >= 6], 
                         key=lambda x: x[1]['avg_score'], reverse=True)[:3]
        
        print("\n🚀 EMERGING OPPORTUNITIES:")
        for i, (interest, data) in enumerate(emerging):
            print(f"{i+1}. {interest}: {data['avg_score']:.1f}/9 avg score, {data['users']:,} users")
        
        # Mass market interests (high volume)
        mass_market = sorted(insights.items(), key=lambda x: x[1]['users'], reverse=True)[:5]
        
        print("\n👥 MASS MARKET INTERESTS:")
        for i, (interest, data) in enumerate(mass_market):
            print(f"{i+1}. {interest}: {data['users']:,} users (avg: {data['avg_score']:.1f}/9)")

def main():
    """Main function to run all analyses"""
    print("=== USER DEMOGRAPHICS AND BEHAVIOR ANALYSIS ===")
    print("Loading and analyzing Data Axle enriched user data...\n")
    
    # Load data
    df = load_and_clean_data()
    
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Total columns: {len(df.columns)}")
    print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Create visualizations
    try:
        create_summary_dashboard(df)
        create_geographic_analysis(df)
        create_demographic_analysis(df)
        create_financial_analysis(df)
        create_interests_analysis(df)
        create_business_insights_report(df)
        
        print("\n✅ Analysis complete! Generated visualizations:")
        print("- summary_dashboard.png: High-level business overview")
        print("- geographic_analysis.png: User geographic distribution")  
        print("- demographic_analysis.png: Age, gender, education demographics")
        print("- financial_analysis.png: Income and wealth analysis")
        print("- interests_analysis.png: User interests with intensity scores (1-9)")
        
        print("\n📊 All visualizations have been saved as high-resolution PNG files.")
        print("🎯 Business insights show high-value segments and opportunities!")
        print("\nThese insights will help your business team understand user engagement intensity,")
        print("not just participation, for more targeted marketing and product development!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check your data file and column names.")

if __name__ == "__main__":
    main() 
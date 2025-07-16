#!/usr/bin/env python3
"""
User Demographics Analysis and Visualization Script
Analyzes user data to help business teams understand their customer base
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_clean_data():
    """Load and clean the user data"""
    print("Loading data...")
    
    # Load customer emails
    emails_df = pd.read_csv('customer_emails.csv')
    total_emails = len(emails_df)
    
    # Load enriched data
    data_df = pd.read_csv('data_axle_results.csv', low_memory=False)
    
    print(f"Total emails in database: {total_emails:,}")
    print(f"Total records with data: {len(data_df):,}")
    
    # Clean the data - remove rows where key demographic fields are empty
    key_fields = [
        'data.document.attributes.first_name',
        'data.document.attributes.gender',
        'data.document.attributes.state',
        'data.document.attributes.family.estimated_income'
    ]
    
    # Count non-null values for each key field
    initial_count = len(data_df)
    
    # Remove rows where all key fields are null
    data_df_clean = data_df.dropna(subset=key_fields, how='all')
    
    # Additional cleaning - remove rows with obviously incomplete data
    data_df_clean = data_df_clean[
        (data_df_clean['data.document.attributes.first_name'].notna()) |
        (data_df_clean['data.document.attributes.gender'].notna()) |
        (data_df_clean['data.document.attributes.state'].notna())
    ]
    
    records_analyzed = len(data_df_clean)
    
    print(f"Records with meaningful data for analysis: {records_analyzed:,}")
    print(f"Records excluded from analysis: {initial_count - records_analyzed:,}")
    print(f"Analysis coverage: {(records_analyzed/total_emails)*100:.1f}% of total emails")
    
    return data_df_clean, total_emails, initial_count, records_analyzed

def create_summary_stats(df, total_emails, initial_count, records_analyzed):
    """Create summary statistics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Data coverage summary
    categories = ['Total Emails', 'Raw Data Records', 'Analyzed Records']
    counts = [total_emails, initial_count, records_analyzed]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    ax1.bar(categories, counts, color=colors)
    ax1.set_title('Data Coverage Summary', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Records')
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts)*0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # Gender distribution
    gender_data = df['data.document.attributes.gender'].value_counts().dropna()
    if not gender_data.empty:
        ax2.pie(gender_data.values, labels=gender_data.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Gender Distribution', fontsize=14, fontweight='bold')
    
    # Age distribution (estimated from family data)
    income_data = df['data.document.attributes.family.estimated_income'].dropna()
    if not income_data.empty:
        ax3.hist(income_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Estimated Income Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Estimated Income ($)')
        ax3.set_ylabel('Number of Users')
    
    # Home ownership
    home_owner_data = df['data.document.attributes.family.estimated_home_owner'].value_counts().dropna()
    if not home_owner_data.empty:
        ax4.pie(home_owner_data.values, labels=['Renter', 'Home Owner'], autopct='%1.1f%%', startangle=90)
        ax4.set_title('Home Ownership Status', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('user_summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_geographic_analysis(df):
    """Analyze geographic distribution of users"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # State distribution (top 15)
    state_data = df['data.document.attributes.state'].value_counts().head(15)
    if not state_data.empty:
        state_data.plot(kind='bar', ax=ax1, color='lightcoral')
        ax1.set_title('Top 15 States by User Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('State')
        ax1.set_ylabel('Number of Users')
        ax1.tick_params(axis='x', rotation=45)
    
    # City distribution (top 10)
    city_data = df['data.document.attributes.city'].value_counts().head(10)
    if not city_data.empty:
        city_data.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_title('Top 10 Cities by User Count', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Users')
    
    # CBSA (Metropolitan areas) distribution
    cbsa_data = df['data.document.attributes.cbsa_level'].value_counts().dropna()
    if not cbsa_data.empty:
        ax3.pie(cbsa_data.values, labels=cbsa_data.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Metropolitan vs Non-Metropolitan Areas', fontsize=14, fontweight='bold')
    
    # Income by state (top 10 states)
    top_states = df['data.document.attributes.state'].value_counts().head(10).index
    state_income = df[df['data.document.attributes.state'].isin(top_states)].groupby('data.document.attributes.state')['data.document.attributes.family.estimated_income'].mean().sort_values(ascending=False)
    if not state_income.empty:
        state_income.plot(kind='bar', ax=ax4, color='orange')
        ax4.set_title('Average Income by State (Top 10 States)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('State')
        ax4.set_ylabel('Average Estimated Income ($)')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('geographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_demographic_analysis(df):
    """Analyze demographic characteristics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Education level distribution
    education_data = df['data.document.attributes.family.estimated_education_level'].value_counts().dropna()
    if not education_data.empty:
        education_data.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Education Level Distribution', fontsize=14, fontweight='bold')
    
    # Family size distribution
    family_size = df['data.document.attributes.family.member_count'].value_counts().dropna().sort_index()
    if not family_size.empty:
        family_size.plot(kind='bar', ax=ax2, color='lightblue')
        ax2.set_title('Family Size Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Family Members')
        ax2.set_ylabel('Number of Users')
    
    # Marital status
    marital_data = df['data.document.attributes.estimated_married'].value_counts().dropna()
    if not marital_data.empty:
        labels = ['Single', 'Married'] if len(marital_data) == 2 else marital_data.index
        ax3.pie(marital_data.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Marital Status Distribution', fontsize=14, fontweight='bold')
    
    # Voter registration status
    voter_data = df['data.document.attributes.registered_voter'].value_counts().dropna()
    if not voter_data.empty:
        labels = ['Not Registered', 'Registered Voter'] if len(voter_data) == 2 else voter_data.index
        ax4.pie(voter_data.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Voter Registration Status', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_financial_analysis(df):
    """Analyze financial characteristics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Income distribution by ranges
    income_data = df['data.document.attributes.family.estimated_income'].dropna()
    if not income_data.empty:
        income_ranges = pd.cut(income_data, bins=[0, 25000, 50000, 75000, 100000, 150000, float('inf')], 
                              labels=['<$25K', '$25K-$50K', '$50K-$75K', '$75K-$100K', '$100K-$150K', '$150K+'])
        income_ranges.value_counts().plot(kind='bar', ax=ax1, color='gold')
        ax1.set_title('Income Distribution by Ranges', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Income Range')
        ax1.set_ylabel('Number of Users')
        ax1.tick_params(axis='x', rotation=45)
    
    # Credit card usage
    cc_count = df['data.document.attributes.credit_card_count'].dropna()
    if not cc_count.empty:
        cc_count.value_counts().sort_index().plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Credit Card Count Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Credit Cards')
        ax2.set_ylabel('Number of Users')
    
    # Liquid assets distribution
    liquid_assets = df['data.document.attributes.family.estimated_liquid_assets[0]'].dropna()
    if not liquid_assets.empty:
        ax3.hist(liquid_assets, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Estimated Liquid Assets Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Estimated Liquid Assets ($)')
        ax3.set_ylabel('Number of Users')
    
    # Investment potential
    investor_data = df['data.document.attributes.family.potential_investor'].value_counts().dropna()
    if not investor_data.empty:
        labels = ['Not Investor', 'Potential Investor'] if len(investor_data) == 2 else investor_data.index
        ax4.pie(investor_data.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Investment Potential', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('financial_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interests_analysis(df):
    """Analyze user interests and behaviors"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Lifestyle segments
    lifestyle_data = df['data.document.attributes.lifestyle_segment'].value_counts().head(10)
    if not lifestyle_data.empty:
        lifestyle_data.plot(kind='barh', ax=ax1, color='purple')
        ax1.set_title('Top 10 Lifestyle Segments', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Users')
    
    # Digital platform usage
    digital_platforms = df['data.document.attributes.digital_platforms_count'].dropna()
    if not digital_platforms.empty:
        digital_platforms.value_counts().sort_index().plot(kind='bar', ax=ax2, color='cyan')
        ax2.set_title('Digital Platforms Usage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Digital Platforms')
        ax2.set_ylabel('Number of Users')
    
    # Top interests (sample from available interest columns)
    interest_columns = [col for col in df.columns if 'interests.' in col and col.endswith(('travel', 'sports', 'cooking', 'technology', 'fitness', 'music'))]
    if interest_columns:
        interest_counts = {}
        for col in interest_columns[:10]:  # Limit to 10 interests
            interest_name = col.split('interests.')[-1]
            count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].value_counts().get(True, 0)
            if count > 0:
                interest_counts[interest_name] = count
        
        if interest_counts:
            interests_df = pd.Series(interest_counts).sort_values(ascending=False)
            interests_df.plot(kind='bar', ax=ax3, color='orange')
            ax3.set_title('Popular User Interests', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Interest Category')
            ax3.set_ylabel('Number of Users')
            ax3.tick_params(axis='x', rotation=45)
    
    # Political affiliation
    political_data = df['data.document.attributes.political_party_affiliation'].value_counts().dropna()
    if not political_data.empty:
        political_data.plot(kind='pie', ax=ax4, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Political Party Affiliation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('interests_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_business_insights(df, total_emails, records_analyzed):
    """Generate key business insights"""
    print("\n" + "="*60)
    print("KEY BUSINESS INSIGHTS")
    print("="*60)
    
    # Data coverage
    coverage_rate = (records_analyzed / total_emails) * 100
    print(f"📊 Data Coverage: {coverage_rate:.1f}% of email addresses have enriched data")
    
    # Geographic insights
    top_states = df['data.document.attributes.state'].value_counts().head(5)
    print(f"\n🗺️  Top 5 Markets by User Count:")
    for state, count in top_states.items():
        percentage = (count / records_analyzed) * 100
        print(f"   • {state}: {count:,} users ({percentage:.1f}%)")
    
    # Demographic insights
    gender_dist = df['data.document.attributes.gender'].value_counts(normalize=True) * 100
    if not gender_dist.empty:
        print(f"\n👥 Gender Distribution:")
        for gender, pct in gender_dist.items():
            print(f"   • {gender}: {pct:.1f}%")
    
    # Income insights
    avg_income = df['data.document.attributes.family.estimated_income'].mean()
    median_income = df['data.document.attributes.family.estimated_income'].median()
    if not pd.isna(avg_income):
        print(f"\n💰 Income Profile:")
        print(f"   • Average estimated income: ${avg_income:,.0f}")
        print(f"   • Median estimated income: ${median_income:,.0f}")
    
    # Home ownership
    home_owners = df['data.document.attributes.family.estimated_home_owner'].value_counts(normalize=True) * 100
    if True in home_owners:
        print(f"   • Home ownership rate: {home_owners[True]:.1f}%")
    
    # Education
    education_dist = df['data.document.attributes.family.estimated_education_level'].value_counts()
    if not education_dist.empty:
        top_education = education_dist.index[0]
        pct = (education_dist.iloc[0] / education_dist.sum()) * 100
        print(f"\n🎓 Education: Most common level is '{top_education}' ({pct:.1f}%)")
    
    # Investment potential
    investors = df['data.document.attributes.family.potential_investor'].value_counts(normalize=True) * 100
    if True in investors:
        print(f"\n📈 Investment Potential: {investors[True]:.1f}% are potential investors")
    
    print("\n" + "="*60)

def main():
    """Main analysis function"""
    print("🔍 CUSTOMER DEMOGRAPHICS ANALYSIS")
    print("="*50)
    
    # Load and clean data
    df, total_emails, initial_count, records_analyzed = load_and_clean_data()
    
    if records_analyzed == 0:
        print("❌ No meaningful data found for analysis.")
        return
    
    print(f"\n✅ Analysis ready! Processing {records_analyzed:,} records...")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    
    print("Creating summary statistics...")
    create_summary_stats(df, total_emails, initial_count, records_analyzed)
    
    print("Creating geographic analysis...")
    create_geographic_analysis(df)
    
    print("Creating demographic analysis...")
    create_demographic_analysis(df)
    
    print("Creating financial analysis...")
    create_financial_analysis(df)
    
    print("Creating interests analysis...")
    create_interests_analysis(df)
    
    # Generate insights
    generate_business_insights(df, total_emails, records_analyzed)
    
    print(f"\n✅ Analysis complete! Generated 5 visualization files:")
    print("   • user_summary_stats.png")
    print("   • geographic_analysis.png") 
    print("   • demographic_analysis.png")
    print("   • financial_analysis.png")
    print("   • interests_analysis.png")

if __name__ == "__main__":
    main() 
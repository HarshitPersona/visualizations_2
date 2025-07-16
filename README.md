# Customer Demographics Analysis

This repository contains tools to analyze customer demographics from email data and create business insights visualizations.

## 📊 Overview

The analysis processes customer email data and enriched demographic information to provide actionable business insights about your user base.

### Data Coverage
- **Total Emails:** 43,752
- **Records Analyzed:** 2,976 (6.8% coverage)
- **Excluded Records:** 3,424 (incomplete data)

## 🚀 Quick Start

### Prerequisites
```bash
pip3 install pandas matplotlib seaborn numpy
```

### Running the Analysis
```bash
python3 analyze_users.py
```

### Viewing Results
Open `dashboard.html` in your web browser to view the complete analysis dashboard.

## 📈 Generated Outputs

The script generates 5 visualization files:

1. **`user_summary_stats.png`** - Data coverage and basic demographics
2. **`geographic_analysis.png`** - Geographic distribution of users
3. **`demographic_analysis.png`** - Age, education, family structure
4. **`financial_analysis.png`** - Income, assets, credit usage
5. **`interests_analysis.png`** - User interests and lifestyle segments

## 🔍 Key Insights

### Demographics
- **Gender:** 58.7% Female, 41.3% Male
- **Income:** $130,084 average, $97,000 median
- **Education:** Associates degree most common (43.8%)
- **Home Ownership:** 78.3% own their homes

### Geographic Distribution
- **Top Markets:** California (13.8%), Texas (8.7%), New York (6.9%)
- **Coverage:** Primarily metropolitan areas

### Financial Profile
- **Investment Potential:** 91.6% are potential investors
- **High Income:** Above-average income levels
- **Asset Holdings:** Significant liquid assets

## 💡 Business Recommendations

### Target Market Focus
- **Primary Markets:** California, Texas, New York
- **Demographics:** Female-majority, high-income audience
- **Financial Services:** Strong investment potential

### Marketing Opportunities
- Home improvement and lifestyle products
- Financial services and investment products
- Geographic expansion in underrepresented markets
- Data enrichment to increase coverage beyond 6.8%

## 📁 File Structure

```
├── customer_emails.csv          # Raw email database
├── data_axle_results.csv       # Enriched demographic data
├── analyze_users.py            # Main analysis script
├── dashboard.html              # Interactive dashboard
├── user_summary_stats.png      # Summary visualizations
├── geographic_analysis.png     # Geographic analysis
├── demographic_analysis.png    # Demographic breakdown
├── financial_analysis.png      # Financial characteristics
├── interests_analysis.png      # Interests and lifestyle
└── README.md                   # This file
```

## 🔧 Customization

The analysis script can be modified to:
- Add new demographic fields
- Change geographic groupings
- Adjust income ranges
- Include additional visualizations
- Export data to different formats

## 📞 Usage Notes

- Empty or incomplete records are automatically filtered out
- The script tracks total vs. analyzed record counts
- All visualizations are saved as high-resolution PNG files
- The dashboard provides an executive-level overview
- Analysis focuses on actionable business insights 
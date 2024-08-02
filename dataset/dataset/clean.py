import pandas as pd

# Load the datasets
try:
    primary_df = pd.read_excel(r'C:\Users\divya\OneDrive\Desktop\smart health diagnosis assistant\dataset\dataset\PCOS_data_without_infertility.xlsx', sheet_name='Full_new')
    second_df = pd.read_csv(r'C:\Users\divya\OneDrive\Desktop\smart health diagnosis assistant\dataset\dataset\PCOS_infertility.csv')
except PermissionError as e:
    print(f"Error loading file: {e}")
    raise

# Merge datasets
try:
    data = pd.merge(primary_df, second_df, on='Patient File No.', suffixes=('', '_y'), how='left')
except TypeError as e:
    print(f"Error in merge: {e}")
    raise

# Drop unnecessary columns
data = data.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y', '  I   beta-HCG(mIU/mL)_y', 'II    beta-HCG(mIU/mL)_y', 'AMH(ng/mL)_y'], axis=1)

# Handle missing values in features
data = data.dropna()  # Or use data.fillna() for imputation if needed

# Encode categorical variables if needed
data = pd.get_dummies(data, drop_first=True)

# Save cleaned data to CSV
cleaned_file_path = r'C:\Users\divya\OneDrive\Desktop\smart health diagnosis assistant\dataset\cleaned_pcos_data.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")

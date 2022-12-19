import pandas as pd

demo = pd.read_csv('NY-clean-demographics')
crime = pd.read_csv('NY-crime-data.csv')
realtor = pd.read_csv('NY-realtor-data.csv')

crime_demo = pd.merge(demo, crime, how="left", left_on="county", right_on="county_name", suffixes=("_demo", "_crime"))
full = pd.merge(realtor, crime_demo, how="left", left_on="zip_code", right_on="zipcode", suffixes=("_real", "_democrime"))

crime_demo.to_csv("crime_demo", index=False)
full.to_csv("NY_full_dataset", index=False)

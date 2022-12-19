import csv

desired_columns = dict()
desired_features = ["major_city",
"county",
"radius_in_miles",
"zipcode",
"unemployment_civilian_labor_force_2020",
"unemployment_employed_2020",
"unemployment_unemployed_2020",
"unemployment_unemployment_rate_2020",
"population_estimate_2019",
"population_change_2019",
"population_births_2019",
"population_deaths_2019",
"population_net_international_migration_2019",
"population_net_domestic_migration_2019",
"population_net_migration_2019",
"unemployment_median_household_income_2019",
"education_percent_of_adults_with_less_than_a_high_school_diploma_2015-19",
"education_percent_of_adults_with_a_high_school_diploma_only_2015-19",
"education_percent_of_adults_completing_some_college_or_associate's_degree_2015-19",
"education_percent_of_adults_with_a_bachelor's_degree_or_higher_2015-19",
"population_by_age_total_Under_5_2019",
"population_by_age_total_5_13_2019",
"population_by_age_total_14_17_2019",
"population_by_age_total_18_24_2019",
"population_by_age_total_16_over_2019",
"population_by_age_total_18_over_2019",
"population_by_age_total_15_44_2019",
"population_by_age_total_25_44_2019",
"population_by_age_total_45_64_2019",
"population_by_age_total_65_over_2019"]

#{'major_city': 1, 'county': 3, 'radius_in_miles': 8, 'zipcode': 11, 'population_estimate_2019': 40, 'population_change_2019': 50, 'population_births_2019': 60, 'population_deaths_2019': 70, 'population_net_international_migration_2019': 90, 'population_net_domestic_migration_2019': 100, 'population_net_migration_2019': 110, 'unemployment_civilian_labor_force_2020': 267, 'unemployment_employed_2020': 268, 'unemployment_unemployed_2020': 269, 'unemployment_unemployment_rate_2020': 270, 'unemployment_median_household_income_2019': 271, 'education_percent_of_adults_with_less_than_a_high_school_diploma_2015-19': 309, 'education_percent_of_adults_with_a_high_school_diploma_only_2015-19': 310, "education_percent_of_adults_completing_some_college_or_associate's_degree_2015-19": 311, "education_percent_of_adults_with_a_bachelor's_degree_or_higher_2015-19": 312, 'population_by_age_total_Under_5_2019': 1216, 'population_by_age_total_5_13_2019': 1219, 'population_by_age_total_14_17_2019': 1222, 'population_by_age_total_18_24_2019': 1225, 'population_by_age_total_16_over_2019': 1228, 'population_by_age_total_18_over_2019': 1231, 'population_by_age_total_15_44_2019': 1234, 'population_by_age_total_25_44_2019': 1237, 'population_by_age_total_45_64_2019': 1240, 'population_by_age_total_65_over_2019': 1243}


with open('NY-demographics.csv', 'r') as infile, open('NY-clean-demographics','w') as outfile:
    reader = csv.reader(infile, delimiter=',')
    writer = csv.writer(outfile, delimiter=',')
    headers = next(reader)
    for j,i in enumerate(headers):
        if i in desired_features and i not in desired_columns.keys():
            desired_columns[i] = j
    #print(list(desired_columns.keys())) # for list above
    writer.writerow(list(desired_columns.keys()))
    for row in reader:
        f_row = [row[1], row[3], row[8], row[11], row[40], row[50], row[60], row[70], row[90], row[100], row[110], row[267], row[268], row[269], row[270], row[271], row[309], row[310], row[311], row[312], row[1216], row[1219], row[1222], row[1225], row[1228], row[1231], row[1234], row[1237], row[1240], row[1243]]
        writer.writerow(f_row)

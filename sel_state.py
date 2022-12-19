import csv
with open('us_county_demographics.csv') as infile, open('NY-demographics.csv', 'w') as outfile:
    reader = csv.reader(infile, delimiter=',')
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(next(reader))
    for row in reader:
        if row[4] == 'NY':
            writer.writerow(row)

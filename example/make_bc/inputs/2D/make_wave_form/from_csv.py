import csv

i_csv = "inlet_original.csv"
o_csv = "inlet.csv"

with open(i_csv) as f:
    reader = csv.reader(f)
    data = [[float(x) for x in row] for row in reader]

cycle = 4

data_ex = []
data_ex.extend(data)

for shift in range(1, cycle):
    for row in data:
        data_ex.append([row[0] + shift, row[1]])

with open(o_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data_ex)

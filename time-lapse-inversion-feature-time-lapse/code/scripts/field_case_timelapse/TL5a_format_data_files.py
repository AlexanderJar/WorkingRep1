import pandas as pd
import pygimli as pg

# RST
read_file = open("rst_filtered_t1.data", 'r')
line_number = 0
for line in read_file:
    line_number += 1
    if "# x y z" in line:
        idx1 = line_number
    if "# g s err t valid " in line:
        idx2 = line_number

rst1_sensors = pd.read_csv("rst_filtered_t1.data", sep='\t', skiprows=idx1, names = ["x","y","z"], nrows=idx2-idx1-2)
rst1_data = pd.read_csv("rst_filtered_t1.data", sep='\t', skiprows=idx2, names = ["g","s","err","t","valid"], skipfooter=1)

read_file = open("rst_filtered_t2.data", 'r')
line_number = 0
for line in read_file:
    line_number += 1
    if "# x y z" in line:
        idx1 = line_number
    if "# g s err t valid " in line:
        idx2 = line_number

rst2_sensors = pd.read_csv("rst_filtered_t2.data", sep='\t', skiprows=idx1, names = ["x","y","z"], nrows=idx2-idx1-2)
rst2_data = pd.read_csv("rst_filtered_t2.data", sep='\t', skiprows=idx2, names = ["g","s","err","t","valid"], skipfooter=1)

rst1_data.set_index(list("gs"), inplace=True)
rst2_data.set_index(list("gs"), inplace=True)

rst1_data, rst2_data = rst1_data.align(rst2_data)

rst1_data.reset_index(inplace=True)
rst2_data.reset_index(inplace=True)

for df in rst1_data, rst2_data:
    # Set normal error
    df["err"] = 0.0003
    # Set dummies
    idx = df["t"].isna()
    df["err"][idx] = 999
    df["t"][idx] = 999

if len(rst2_sensors) != len(rst1_sensors):
    print("Check sensors.")

def export_rst(data, sensors, filename):

    f = open(filename, 'w')
    f.write("%d\n" % len(sensors))
    f.write("# ")

    for key in sensors.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in sensors.itertuples(index=False):
        for val in row:
            f.write("%5.3f " % val)
        f.write("\n")
    f.write("%d\n" % len(data))
    f.write("# ")
    
    for key in data.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in data.itertuples(index=False):
        for i, val in enumerate(row):
            if i < 2 or i == 4:
                f.write("%s " % val)
            else:
                f.write("%f " % val)

        f.write("\n")
    f.write("0")
    f.close()

export_rst(rst1_data, rst1_sensors, 'rst_formatted_t1.data')
export_rst(rst2_data, rst2_sensors, 'rst_formatted_t2.data')


# ERT
read_file = open("ert_filtered_t1.data", 'r')
line_number = 0
for line in read_file:
    line_number += 1
    if "# x y z" in line:
        idx1 = line_number
    if "# a b m n err i ip iperr k r rhoa u valid " in line:
        idx2 = line_number

ert1_sensors = pd.read_csv("ert_filtered_t1.data", sep='\t', skiprows=idx1, names = ["x","y","z"], nrows=idx2-idx1-2)
ert1_data = pd.read_csv("ert_filtered_t1.data", sep='\t', skiprows=idx2, names = ["a","b","m","n","err","i","ip","iperr","k","r","rhoa","u","valid"], skipfooter=1)

read_file = open("ert_filtered_t2.data", 'r')
line_number = 0
for line in read_file:
    line_number += 1
    if "# x y z" in line:
        idx1 = line_number
    if "# a b m n err i ip iperr k r rhoa u valid " in line:
        idx2 = line_number

ert2_sensors = pd.read_csv("ert_filtered_t2.data", sep='\t', skiprows=idx1, names = ["x","y","z"], nrows=idx2-idx1-2)
ert2_data = pd.read_csv("ert_filtered_t2.data", sep='\t', skiprows=idx2, names = ["a","b","m","n","err","i","ip","iperr","k","r","rhoa","u","valid"], skipfooter=1)

ert1_data.set_index(list("abmn"), inplace=True)
ert2_data.set_index(list("abmn"), inplace=True)

ert1_data, ert2_data = ert1_data.align(ert2_data)

ert1_data.reset_index(inplace=True)
ert2_data.reset_index(inplace=True)

for df in ert1_data, ert2_data:
    # Set normal error
    df["err"] = 0.0003
    # Set dummies
    idx = df["rhoa"].isna()
    df["err"][idx] = 999
    df["rhoa"][idx] = 999999

if len(ert2_sensors) != len(ert1_sensors):
    print("Check sensors.")

def export_ert(data, sensors, filename):

    f = open(filename, 'w')
    f.write("%d\n" % len(sensors))
    f.write("# ")

    for key in sensors.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in sensors.itertuples(index=False):
        for val in row:
            f.write("%5.3f " % val)
        f.write("\n")
    f.write("%d\n" % len(data))
    f.write("# ")
    
    for key in data.keys():
        f.write("%s " % key)
    f.write("\n")
    for row in data.itertuples(index=False):
        for i, val in enumerate(row):
            if i < 4:
                f.write("%d " % val)
            else:
                f.write("%E " % val)

        f.write("\n")
    f.write("0")
    f.close()

export_ert(ert1_data, ert1_sensors, 'ert_formatted_t1.data')
export_ert(ert2_data, ert2_sensors, 'ert_formatted_t2.data')
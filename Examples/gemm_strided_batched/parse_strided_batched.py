#!/usr/bin/python

import sys
import argparse

# count number of lines in file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("in_filename",help="input filename")
    parser.add_argument("dataset",help="dataset name; like conv_resnet50_fwd_fp32")
    parser.add_argument("datatype",help="datatype, half | float | double")
    args = parser.parse_args()

#   get filename from command line arguments
    input_filename = args.in_filename
    dataset = args.dataset + "_sb"
    datatype = args.datatype

#   open input file and read from it into lines
    try:
        f_in = open(input_filename, 'rb')
    except IOError:
        print "Could not read file: ", input_filename
        sys.exit()
    lines = f_in.readlines()

#   open output file
    f_out=open('./' + input_filename + ".out", 'w+')

#   write tuples
    iterator = 0
    for line in lines:
        iterator = iterator + 1
        f_out.write("gemm_strided_batched_tuple %s_%03d %s" % (dataset,iterator,line))

#   write vector of tuples
    num_lines = file_len(input_filename)
    f_out.write("\nconst vector<gemm_strided_batched_tuple> %s = {\n" % (dataset))
    for i in range(1,num_lines+1):
        f_out.write("%s_%03d, " % (dataset,i))
        if i%4 == 0:
            f_out.write("\n")
    if num_lines%4 != 0:
        f_out.write("\n")
    f_out.write("};\n")

    f_out.write("\nINSTANTIATE_TEST_CASE_P(nightly_%s, gemm_strided_batched_%s, ValuesIn(%s));" % (dataset, datatype, dataset))

if __name__ == "__main__":
    main()

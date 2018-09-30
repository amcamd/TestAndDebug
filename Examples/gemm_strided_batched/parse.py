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
    args = parser.parse_args()
    print "in_filename = ",args.in_filename

#   get filename from command line arguments
    input_filename = sys.argv[1]


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
        f_out.write("gemm_tuple conv_resnet50_fwd_fp32_%03d=%s" % (iterator,line))

#   write vector of tuples
    num_lines = file_len(input_filename)
    f_out.write("\nconst vector<gemm_tuple> deepbench_vec = {\n")
    for i in range(1,num_lines+1):
        f_out.write("gemm_tuple conv_resnet50_fwd_fp32_%03d, " % (i))
        if i%2 == 0:
            f_out.write("\n")
    if num_lines%2 != 0:
        f_out.write("\n")
    f_out.write("};\n")


if __name__ == "__main__":
    main()

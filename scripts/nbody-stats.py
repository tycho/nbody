#!/usr/bin/env python2

# Simple script taking n-body output and printing a CSV of statistical data
# from the results.

import fileinput
import re

line_re = r'^\s*(?P<algorithm>[\w\s]+):\s*(?P<ms>[\d.]+)\s*ms\s*=\s*(?P<interactions>[\d\.x^]+)\s*interactions/s+\s*\(\s*(?P<gflops>[\d.]+)\s*GFLOPS\)(?:\s*\(Rel\. error:\s*(?P<error>[\d\.E+-]+)\))?$'
line_rex = re.compile(line_re)

algs = {}

for line in fileinput.input():
    line = line.rstrip()
    m = line_rex.search(line)
    if m:
        alg = m.group('algorithm')
        gflops = float(m.group('gflops'))
        if alg not in algs:
            algs[alg] = []
        algs[alg].append(gflops)

print "algorithm,min,max,avg,stdev"
for alg, values in algs.items():
    avg = sum(values) / float(len(values))

    # Find standard deviation.
    dif = [ (v - avg) ** 2.0 for v in values ]
    var = sum(dif) / float(len(dif))
    std = var ** 0.5

    print "%s,%f,%f,%f,%f" % (
        alg,
        min(values),
        max(values),
        sum(values)/len(values),
        std,
        )

# vim: set ts=4 sts=4 sw=4 et:

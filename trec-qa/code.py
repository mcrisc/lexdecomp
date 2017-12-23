# -*- coding: utf-8 -*-

f = open('train-filtered.tsv')

lines = f.readlines()

lines = [line.split('\t') for line in lines]